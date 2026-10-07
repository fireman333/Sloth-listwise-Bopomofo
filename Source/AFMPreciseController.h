// AFMPreciseController: precise mode (design D2). Owns the heavy re-ranking
// model (QwenRescorer) and its calibration (PreciseConfig), loads them off the
// main thread while precise mode is enabled, releases them when it is
// disabled, and runs at most one heavy inference at a time on a private
// serial queue for all clients (latest request wins the single pending slot).
//
// Threading: every public method and every state change happens on the main
// thread; the heavy model is only touched on the serial queue. Results are
// handed back on the main thread to the requester that made the request, and
// only if precise mode was not toggled meanwhile (epoch check).
//
// Logs carry state changes and timings only, never readings or candidates.

#import <Foundation/Foundation.h>

#ifdef __cplusplus
#include <string>
#include <vector>
#include "Rescorer/precise_walk.h"
#endif

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, AFMPreciseState) {
    /// Precise mode is disabled; nothing is loaded.
    AFMPreciseStateOff = 0,
    AFMPreciseStateLoading,
    AFMPreciseStateReady,
    /// The model directory, the calibration file or the model file is missing.
    AFMPreciseStateNotFound,
    /// Invalid calibration, calibration alpha != R-walk alpha, or the heavy
    /// model failed to load.
    AFMPreciseStateFailed,
    /// Three consecutive heavy inferences timed out. Sticky for the process,
    /// even across disable / enable.
    AFMPreciseStateTripped,
};

/// Posted on the main thread (object: the controller) whenever `state` changes.
extern NSNotificationName const AFMPreciseControllerStateDidChangeNotification;

/// Test-only heavy scorer: readings and candidate sentences -> one score per
/// candidate (higher is better). Return nil or an empty array to simulate a
/// failure. Runs on the controller's serial queue.
typedef NSArray<NSNumber *> *_Nullable (^AFMPreciseHeavyScoreBlock)(
    NSArray<NSString *> *readings, NSArray<NSString *> *candidates);

@class AFMPreciseRequest;

@protocol AFMPreciseRequester <NSObject>
/// Called on the main thread when the heavy model scored the request and the
/// fused choice is not candidate 0 (the displayed text).
- (void)preciseRequestDidFinish:(AFMPreciseRequest *)request chosenIndex:(NSUInteger)chosenIndex;
@end

/// One background re-ranking request: strings and numbers only (no grid
/// nodes cross threads). Candidate 0 is the displayed text.
@interface AFMPreciseRequest : NSObject
#ifdef __cplusplus
- (instancetype)initWithReadings:(std::vector<std::string>)readings
                      candidates:(std::vector<std::string>)candidates
                         rScores:(std::vector<double>)rScores
                             key:(NSString *)key
                   composingText:(NSString *)composingText
                       requester:(id<AFMPreciseRequester>)requester;
- (const std::vector<std::string> &)readings;
- (const std::vector<std::string> &)candidates;
- (const std::vector<double> &)rScores;
#endif
/// Test-only: same as the C++ initializer.
- (instancetype)initForTestingWithReadings:(NSArray<NSString *> *)readings
                                candidates:(NSArray<NSString *> *)candidates
                                   rScores:(NSArray<NSNumber *> *)rScores
                                       key:(NSString *)key
                             composingText:(NSString *)composingText
                                 requester:(id<AFMPreciseRequester>)requester;
- (instancetype)init NS_UNAVAILABLE;
/// The requester's request key (readings, characters, fixed positions,
/// generations). Never logged.
@property (readonly, nonatomic) NSString *key;
/// The composing buffer shown when the request was made, for the input
/// method controller's check before applying. Never logged.
@property (readonly, nonatomic) NSString *composingText;
@property (readonly, nonatomic, weak, nullable) id<AFMPreciseRequester> requester;
/// The controller epoch the request was submitted in.
@property (readonly, nonatomic) NSUInteger epoch;
@end

@interface AFMPreciseController : NSObject

/// The process-wide controller, created lazily with
/// UserPhraseLocationHelper.qwenModelDirectory. Tests may replace it.
@property (class, nonatomic, strong) AFMPreciseController *sharedController NS_SWIFT_NAME(shared);

/// Replaces the shared controller with one whose model directory does not
/// exist (enabling it ends in NotFound), so precise mode is never ready.
+ (void)resetForTesting;

/// Production controller: <directory>/precise.conf, then <directory>/model.gguf.
- (instancetype)initWithModelDirectory:(nullable NSString *)modelDirectory;
/// Same as the production controller, with the real Qwen model using the
/// given number of CPU threads (production: 4). Diagnostics only.
- (instancetype)initWithModelDirectory:(nullable NSString *)modelDirectory
                 qwenThreadsForTesting:(NSInteger)threads;
/// Same, also overriding the Precise queue's QoS (production: utility).
- (instancetype)initWithModelDirectory:(nullable NSString *)modelDirectory
                 qwenThreadsForTesting:(NSInteger)threads
                    queueQoSForTesting:(qos_class_t)qos;
/// Test controller: reads <directory>/precise.conf like production, then
/// (after loadDelay seconds on the queue) uses scoreBlock as the heavy model
/// instead of loading model.gguf, which need not exist.
- (instancetype)initWithModelDirectory:(nullable NSString *)modelDirectory
             heavyScoreBlockForTesting:(AFMPreciseHeavyScoreBlock)scoreBlock
                             loadDelay:(NSTimeInterval)loadDelay;
- (instancetype)init NS_UNAVAILABLE;

/// Enables (starts a background load) or disables (releases the model and
/// discards pending and in-flight results). Same value again: no-op. The
/// calibration's alpha is checked against Preferences.neuralWalkAlpha.
@property (assign, nonatomic) BOOL enabled;

/// Applies AFMPreciseModeEnabled; if enabled and AFMNeuralWalkAlpha changed
/// since the last load, reloads so the alpha check is redone. Never blocks.
- (void)syncWithPreferences;

/// Queues a request (main thread). It runs now if nothing runs; otherwise it
/// replaces any queued request. Dropped unless the state is Ready.
- (void)submitRequest:(AFMPreciseRequest *)request NS_SWIFT_NAME(submitRequest(_:));

@property (readonly, nonatomic) AFMPreciseState state;
@property (readonly, nonatomic, getter=isReady) BOOL ready;
/// Incremented on every enable / disable / reload / trip. Results from an
/// older epoch are never delivered.
@property (readonly, nonatomic) NSUInteger epoch;
@property (readonly, nonatomic, nullable) NSString *modelDirectory;
/// Wall time of the last completed load attempt, in ms (0 if none yet).
@property (readonly, nonatomic) double lastLoadMilliseconds;
/// Requests accepted by submitRequest: (tests).
@property (readonly, nonatomic) NSUInteger submittedCount;
/// Requests whose heavy inference actually ran (tests).
@property (readonly, nonatomic) NSUInteger scoredCount;
/// Results handed to a requester (tests).
@property (readonly, nonatomic) NSUInteger deliveredCount;
/// Consecutive timed-out inferences.
@property (readonly, nonatomic) NSUInteger timeoutStreak;
/// Loads that finished after precise mode was toggled again and were dropped (tests).
@property (readonly, nonatomic) NSUInteger discardedLoadCount;
/// Main thread: a key event was just handled. No heavy inference starts
/// until startDelay seconds after the last key event (queued requests wait;
/// latest wins).
- (void)noteKeyEvent;
/// Default 0.04 s. Tests may change it.
@property (assign, nonatomic) NSTimeInterval startDelay;
/// Sum and max of heavy inference wall times in ms (tests / diagnostics).
@property (readonly, nonatomic) double totalHeavyMilliseconds;
@property (readonly, nonatomic) double maxHeavyMilliseconds;
/// A heavy inference is running (main-thread view; tests).
@property (readonly, nonatomic, getter=isHeavyRunning) BOOL heavyRunning;
/// Times a start was deferred because of a recent key event (tests).
@property (readonly, nonatomic) NSUInteger deferredStartCount;
/// qos_class_self() observed inside the last heavy-inference block (tests).
@property (readonly, nonatomic) qos_class_t lastHeavyQoSClass;
/// No request running or queued (tests).
@property (readonly, nonatomic, getter=isIdle) BOOL idle;

#ifdef __cplusplus
/// The calibration loaded with the current model; valid only while ready.
- (const McBopomofo::PreciseConfig &)config;
#endif

@end

NS_ASSUME_NONNULL_END
