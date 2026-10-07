// AFMNeuralSelector: owns the R model (SlothLM-E ranker export) used for
// R-walk, loads it off the main thread, and decides per walk whether the
// composing buffer is selected with R-walk or with the stock walk.
//
// Fallback chain: R-walk -> stock walk. Any problem (onnxruntime missing,
// model directory missing, load failure, empty score table, slow inference,
// switch off, alpha 0, > kAFMNeuralMaxSyllables) means the stock
// ReadingGrid::walk() for that walk; typing is never blocked. The caller
// (KeyHandler) only asks for it in Bopomofo mode.
//
// Threading: inference and every state read/write happen on the main thread;
// only model loading runs on a private serial queue, and its result is
// published back on the main queue.

#import <Foundation/Foundation.h>

#ifdef __cplusplus
#import "reading_grid.h"
#include <optional>
#include <string>
#include "Rescorer/precise_walk.h"
#endif

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, AFMNeuralSelectorState) {
    /// Not loaded because the switch was off when loading was last considered.
    AFMNeuralSelectorStateDisabled = 0,
    AFMNeuralSelectorStateLoading,
    AFMNeuralSelectorStateReady,
    /// onnxruntime missing or load failure. Sticky.
    AFMNeuralSelectorStateFailed,
    /// Slow-inference circuit breaker tripped. Sticky for the process.
    AFMNeuralSelectorStateTripped,
    /// No model directory configured, or it does not exist. Sticky.
    AFMNeuralSelectorStateNotFound,
};

/// Posted on the main thread (object: the selector) whenever `state` changes.
extern NSNotificationName const AFMNeuralSelectorStateDidChangeNotification;

#ifdef __cplusplus
/// What the end-of-key-event hook needs for a precise-mode request (design D1
/// steps 1-5): the R N-best that keeps every fixed position, displayed text
/// first, and the calibrated gate on it.
struct AFMPrecisePayload {
    McBopomofo::PreciseCandidateSet candidates;
    McBopomofo::PreciseDecision gate;
};
#endif

/// Readings above this length always use the stock walk (design D10).
extern const NSUInteger kAFMNeuralMaxSyllables;

@class AFMPersonalVersionRequest;

/// Personalized weights of the loaded R (personal-finetune design D11).
typedef NS_ENUM(NSInteger, AFMPersonalModelStatus) {
    /// No personalized version selected: factory weights.
    AFMPersonalModelStatusFactory = 0,
    /// A personalized version is being loaded in the background.
    AFMPersonalModelStatusLoading,
    /// The selected version is in use (loadedPersonalVersion).
    AFMPersonalModelStatusActive,
    /// The selected version could not be loaded: factory weights.
    AFMPersonalModelStatusUnusable,
    /// The selected version was evaluated with another configuration
    /// (factory model, alpha, reading policy, pack): factory weights.
    AFMPersonalModelStatusMismatch,
};

/// Test-only scorer: readings (one syllable each) and, per position, the
/// characters the grid can place there -> per position {character: log p}.
/// Return nil or an empty array to simulate an inference failure.
typedef NSArray<NSDictionary<NSString *, NSNumber *> *> *_Nullable (^AFMNeuralScoreBlock)(
    NSArray<NSString *> *readings, NSArray<NSArray<NSString *> *> *charsAtPositions);

/// Test-only: the score block for a personalized version (nil = load failure,
/// *error set). Runs on the load queue.
typedef AFMNeuralScoreBlock _Nullable (^AFMPersonalScoreBlockProvider)(AFMPersonalVersionRequest *request, NSString *_Nullable *_Nonnull error);

@interface AFMNeuralSelector : NSObject

/// The process-wide selector, created lazily with
/// UserPhraseLocationHelper.rWalkModelDirectory. Tests may replace it.
@property (class, nonatomic, strong) AFMNeuralSelector *sharedSelector NS_SWIFT_NAME(shared);

/// Replaces the shared selector with one that has no model (its first load
/// attempt fails), so every walk is the stock walk.
+ (void)resetForTesting;

/// Production selector loading SlothRescorer from modelDirectory (nil = no
/// model; the first load attempt fails).
- (instancetype)initWithModelDirectory:(nullable NSString *)modelDirectory;
/// Fake model: loading sleeps loadDelay seconds on the load queue, then the
/// selector becomes ready and scores with scoreBlock.
- (instancetype)initWithScoreBlockForTesting:(AFMNeuralScoreBlock)scoreBlock
                                   loadDelay:(NSTimeInterval)loadDelay;
/// Fake model with personalized versions: factory loads use scoreBlock,
/// personalized loads personalBlockProvider (both after loadDelay).
- (instancetype)initWithScoreBlockForTesting:(AFMNeuralScoreBlock)scoreBlock
                       personalBlockProvider:(nullable AFMPersonalScoreBlockProvider)personalBlockProvider
                                   loadDelay:(NSTimeInterval)loadDelay;
- (instancetype)init NS_UNAVAILABLE;

/// Reads AFMNeuralWalkEnabled / AFMNeuralWalkAlpha and, if enabled, starts
/// loading (non-blocking). Call on launch and on every activation.
- (void)syncWithPreferences;
/// Starts the background load if the switch is on and nothing was loaded yet
/// (state disabled). Never blocks.
- (void)startLoadingIfNeeded;

@property (readonly, nonatomic) AFMNeuralSelectorState state;

// MARK: Personalized weights (personal-finetune design D11)
/// The version to use; nil = AFMPersonalTrainingController.shared's
/// activeVersionRequest. Read at every syncWithPreferences (activation): if it
/// differs from the loaded one, a new scorer is loaded in the background
/// while the current one stays in use, then swapped in on main (the old one
/// is freed on the load queue).
@property (copy, nonatomic, nullable) AFMPersonalVersionRequest *_Nullable (^personalVersionProvider)(void);
/// The current evaluation configuration for a load (alpha, reading policy);
/// runs on the load queue. nil = factory SHA-256 and pack version read from
/// modelDirectory (tests replace it).
@property (copy, nonatomic, nullable) NSDictionary *_Nonnull (^currentEvalConfigProvider)(double alpha, BOOL acceptYan);
/// Clearing (design D13 step 2): factory weights at once — a personalized
/// scorer in use is replaced in the background, and every personalized load
/// started before is discarded on completion.
- (void)requestFactoryWeights;
@property (readonly, nonatomic) AFMPersonalModelStatus personalStatus;
/// Version of the loaded personalized weights; 0 = factory.
@property (readonly, nonatomic) NSInteger loadedPersonalVersion;
/// Why the selected version is not used (unusable / mismatch), else nil.
@property (readonly, nonatomic, nullable) NSString *personalError;
/// Testing: loads started, loads discarded on completion, scorers freed.
@property (readonly, nonatomic) NSUInteger loadCount;
@property (readonly, nonatomic) NSUInteger discardedLoadCount;
@property (readonly, nonatomic) NSUInteger freedScorerCount;
/// Testing: whether a background (re)load is in flight.
@property (readonly, nonatomic, getter=isReloading) BOOL reloading;
/// Testing: waits until the load queue is idle.
- (void)waitForLoadQueueForTesting;
@property (assign, nonatomic) BOOL enabled;
@property (assign, nonatomic) double alpha;
@property (readonly, nonatomic, nullable) NSString *modelDirectory;
/// Identifies the loaded model (set when it becomes ready); nil before.
@property (readonly, nonatomic, nullable) NSString *modelIdentifier;
/// Wall time of the last completed load attempt, in ms (0 if none yet).
@property (readonly, nonatomic) double lastLoadMilliseconds;
/// Number of walks that ran a forward pass (cache misses), for tests.
@property (readonly, nonatomic) NSUInteger forwardCount;
/// Number of walks that used neural scores (fresh or cached), for tests.
@property (readonly, nonatomic) NSUInteger neuralWalkCount;
/// Number of breaker measurements recorded (one per cache-miss walk), for tests.
@property (readonly, nonatomic) NSUInteger breakerRecordCount;
/// Test-only: measurements are still counted but never trip the breaker.
@property (assign, nonatomic) BOOL circuitBreakerDisabledForTesting;
/// Number of precise payloads computed (N-best + gate), for tests.
@property (readonly, nonatomic) NSUInteger precisePayloadCount;

/// Records a breaker measurement deferred by
/// walkGrid:deferBreakerRecording:usedNeural: (at most one is pending),
/// adding extraMilliseconds (N-best + gate time). No-op if none is pending.
- (void)flushPendingBreakerRecordingAddingMilliseconds:(double)extraMilliseconds;

#ifdef __cplusplus
/// Walks the grid for the composing buffer: R-walk when the model is ready,
/// enabled, alpha != 0, not tripped and the grid holds
/// 1...kAFMNeuralMaxSyllables readings; the stock walk otherwise.
- (Formosa::Gramambular2::ReadingGrid::WalkResult)walkGrid:(Formosa::Gramambular2::ReadingGrid &)grid;

/// Same, reporting whether neural scores were used. With deferBreakerRecording
/// (precise mode ready), a cache-miss measurement is kept pending instead of
/// recorded, so the end-of-key-event hook can add the N-best and gate time;
/// a pending measurement is always recorded first by the next walk or hook.
- (Formosa::Gramambular2::ReadingGrid::WalkResult)walkGrid:(Formosa::Gramambular2::ReadingGrid &)grid
                                      deferBreakerRecording:(BOOL)deferBreakerRecording
                                                 usedNeural:(nullable BOOL *)usedNeural;

/// The precise payload for the grid on the cached score table (no forward
/// pass): N-best (config.k, config.alpha), minus candidates that change a
/// fixed position, with displayedText moved to rank 0, and the gate. Returns
/// std::nullopt when R-walk is not usable, the table is not cached, fewer than
/// two candidates remain, or displayedText is not R's rank 0 (or tied with it).
- (std::optional<AFMPrecisePayload>)precisePayloadForGrid:(Formosa::Gramambular2::ReadingGrid &)grid
                                            displayedText:(const std::string &)displayedText
                                                   config:(const McBopomofo::PreciseConfig &)config;
#endif

@end

NS_ASSUME_NONNULL_END
