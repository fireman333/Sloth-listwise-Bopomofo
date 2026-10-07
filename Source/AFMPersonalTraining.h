// AFMPersonalTrainingController: personalized fine-tuning of R (openspec
// change personal-finetune, design D1, D6, D7, D13, D15, D16). Spawns the
// bundled AFMPersonalTrainer helper, watches it, adopts its result into
// memory/r-personal/v<N>/ and keeps manifest.json (the IME is the single
// writer of adoption state), and tells AFMNeuralSelector which version to
// load at the next activation.
//
// Threading: the API is main-thread only. Every file under memory/r-personal/
// is read and written on the learning store queue; results come back on main.
// Every queued block and every helper result captures the personalization
// generation (bumped only by clearAll) and is discarded if it changed.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Posted on the main thread (object: the controller) whenever its state
/// (running, eligible count, manifest, last run) changes.
extern NSNotificationName const AFMPersonalTrainingControllerDidChangeNotification;

/// Folder and file names under the learning data directory (design D13).
extern NSString *const AFMPersonalDirectoryName;  // r-personal

typedef NS_ENUM(NSInteger, AFMPersonalStartResult) {
    AFMPersonalStartResultStarted = 0,
    AFMPersonalStartResultLearningOff,
    AFMPersonalStartResultNotEnoughData,
    AFMPersonalStartResultAlreadyRunning,
    AFMPersonalStartResultPackMissing,
    AFMPersonalStartResultHelperMissing,
    AFMPersonalStartResultSpawnFailed,
    AFMPersonalStartResultClearing,
};

/// Why an automatic run does not start now (design D6), in check order.
typedef NS_ENUM(NSInteger, AFMPersonalBlockReason) {
    AFMPersonalBlockReasonNone = 0,
    AFMPersonalBlockReasonLearningOff,
    AFMPersonalBlockReasonPaused,
    AFMPersonalBlockReasonFailurePaused,
    AFMPersonalBlockReasonRunning,
    AFMPersonalBlockReasonPackMissing,
    AFMPersonalBlockReasonNotEnoughData,
    AFMPersonalBlockReasonTooSoon,
    AFMPersonalBlockReasonNotEnoughNew,
    AFMPersonalBlockReasonOnBattery,
    AFMPersonalBlockReasonLowPowerMode,
    AFMPersonalBlockReasonThermal,
    AFMPersonalBlockReasonNotIdle,
};

/// Power / thermal / idle probes (injectable for tests, design D16).
@interface AFMPersonalSystemProbes : NSObject
@property (copy, nonatomic) BOOL (^onACPower)(void);
@property (copy, nonatomic) BOOL (^lowPowerMode)(void);
/// thermalState serious or critical.
@property (copy, nonatomic) BOOL (^thermalPressure)(void);
/// Seconds since the last keyboard or mouse input (HIDIdleTime).
@property (copy, nonatomic) double (^idleSeconds)(void);
/// The real system probes.
+ (instancetype)systemProbes;
@end

/// The personalized version the selector should use (design D11), with the
/// personalization generation it was requested in.
@interface AFMPersonalVersionRequest : NSObject
@property (readonly, nonatomic) NSInteger version;
/// memory/r-personal/v<N>
@property (readonly, nonatomic) NSString *directory;
@property (readonly, nonatomic) NSString *runIdentifier;
@property (readonly, nonatomic) NSDictionary *evalConfig;
@property (readonly, nonatomic) uint64_t generation;
/// Creation time of the version (seconds since 1970).
@property (readonly, nonatomic) double created;
- (instancetype)initWithVersion:(NSInteger)version
                      directory:(NSString *)directory
                  runIdentifier:(NSString *)runIdentifier
                     evalConfig:(NSDictionary *)evalConfig
                     generation:(uint64_t)generation
                        created:(double)created NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
@end

/// The evaluation configuration of the running IME (design D11 / Open
/// Question 20): factory SHA-256 of models/r-walk/onnx/model_quantized.onnx,
/// alpha, reading policy, pack and calibration version. Hashing is cached by
/// size and mtime; call off the main thread when not cached.
FOUNDATION_EXPORT NSString *_Nullable AFMFactoryModelSHA256(NSString *modelDirectory);
FOUNDATION_EXPORT NSString *_Nullable AFMPersonalPackVersion(NSString *modelDirectory);
FOUNDATION_EXPORT NSDictionary *AFMCurrentEvalConfig(NSString *_Nullable factorySHA, NSString *_Nullable packVersion, double alpha, BOOL acceptYan);
/// Whether a version's eval_config equals the current one (alpha within 1e-9).
FOUNDATION_EXPORT BOOL AFMEvalConfigMatches(NSDictionary *_Nullable versionConfig, NSDictionary *_Nullable currentConfig);

/// Reads "run" from a precise-r.conf overlay (nil if missing / unreadable).
FOUNDATION_EXPORT NSString *_Nullable AFMPreciseOverlayRun(NSString *path);

/// Testing: writes a weights.bin of `count` 2x2 tensors bound to the factory
/// SHA-256 `factorySHA` (hex).
FOUNDATION_EXPORT BOOL AFMWriteFakeWeightsFileForTesting(NSString *path, NSString *factorySHA, NSInteger count);
/// Testing: writes the 84 factory linears of modelDirectory as a weights.bin,
/// every scale multiplied by scaleFactor (1 = the factory weights).
FOUNDATION_EXPORT BOOL AFMWriteFactoryWeightsFileForTesting(NSString *modelDirectory, NSString *path, double scaleFactor);

@interface AFMPersonalTrainingController : NSObject

@property (class, nonatomic, strong) AFMPersonalTrainingController *sharedController NS_SWIFT_NAME(shared);
/// Replaces the shared controller with one on the default directories, no
/// helper path, no scheduler (tests).
+ (void)resetForTesting;

/// memoryDirectory = <root>/memory, modelDirectory = <root>/models/r-walk,
/// helperPath = the bundled AFMPersonalTrainer (nil: none).
- (instancetype)initWithMemoryDirectory:(NSString *)memoryDirectory
                         modelDirectory:(NSString *)modelDirectory
                             helperPath:(nullable NSString *)helperPath NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@property (readonly, nonatomic) NSString *memoryDirectory;
@property (readonly, nonatomic) NSString *modelDirectory;
@property (readonly, nonatomic) NSString *personalDirectory;  // memory/r-personal
@property (copy, nonatomic, nullable) NSString *helperPath;

// MARK: Injectable configuration (design D16)
/// Wall clock (seconds since 1970); nil = system clock.
@property (copy, nonatomic, nullable) double (^clock)(void);
@property (strong, nonatomic) AFMPersonalSystemProbes *probes;
/// Learning switch; nil = Preferences.learningEnabled.
@property (copy, nonatomic, nullable) BOOL (^learningEnabledCheck)(void);
@property (assign, nonatomic) NSTimeInterval runTimeLimit;        // 600 s
@property (assign, nonatomic) NSTimeInterval killGracePeriod;     // 5 s
@property (assign, nonatomic) NSInteger minimumEligible;          // 1,000
@property (assign, nonatomic) NSInteger minimumNewEligible;       // 200
@property (assign, nonatomic) NSTimeInterval automaticInterval;   // 24 h
@property (assign, nonatomic) NSTimeInterval minimumIdle;         // 10 min
@property (assign, nonatomic) NSInteger helperThreads;            // 4
@property (assign, nonatomic) NSInteger failurePauseCount;        // 3

// MARK: State (main thread)
@property (readonly, nonatomic) uint64_t generation;
/// Eligible segments from the last pool scan; -1 before the first scan.
@property (readonly, nonatomic) NSInteger eligibleCount;
@property (readonly, nonatomic, getter=isRunning) BOOL running;
@property (readonly, nonatomic, nullable) NSString *currentRunIdentifier;
@property (readonly, nonatomic) BOOL currentRunIsAutomatic;
@property (readonly, nonatomic) BOOL packAvailable;
@property (readonly, nonatomic) BOOL paused;
@property (readonly, nonatomic) BOOL failurePaused;
@property (readonly, nonatomic) NSInteger consecutiveFailures;
@property (readonly, nonatomic, nullable) NSNumber *activeVersion;
@property (readonly, nonatomic, nullable) NSNumber *previousVersion;
/// Creation time (seconds since 1970) of a kept version, or 0.
- (double)createdTimeOfVersion:(NSInteger)version;
/// The last finished run: run, time, mode, result, exit_code, eligible,
/// delta_personal_pt, delta_general_pt, p, eval_config (no text).
@property (readonly, nonatomic, nullable) NSDictionary *lastRun;
/// The factory SHA-256 and pack version found by the last refresh.
@property (readonly, nonatomic, nullable) NSString *factorySHA;
@property (readonly, nonatomic, nullable) NSString *packVersion;
/// Current evaluation configuration (factory, pack, live alpha / policy).
- (NSDictionary *)currentEvalConfig;

// MARK: Actions (main thread)
/// Off-main: factory SHA and pack, factory-update handling (D13), crashed
/// staging cleanup, pool scan for the eligible count (on the store queue).
/// Completion on main.
- (void)refreshWithCompletion:(nullable dispatch_block_t)completion;
/// Pool scan only (coalesced; skipped when the pool size is unchanged).
- (void)refreshEligibleCountWithCompletion:(nullable dispatch_block_t)completion;

/// 「立即學習」: learning on, eligible >= minimum, not running; ignores
/// power, idle, 24 h and new-segment rules; clears both pauses.
- (AFMPersonalStartResult)startManualRun;
/// Automatic rules (design D6).
- (AFMPersonalBlockReason)automaticBlockReason;
/// Scheduler body: spawns an automatic run when nothing blocks it.
- (BOOL)runAutomaticAttemptIfPossible;
/// SIGTERM, then SIGKILL after killGracePeriod. Result "cancelled".
- (void)cancelRun;

/// 「改回出廠模型」: active = none, previous = the old active, paused.
- (void)revertToFactory;
/// 「改用上一版」: swap active and previous.
- (void)usePreviousVersion;

/// Clearing (design D13): bump the generation, tell the selector to use the
/// factory weights, kill a running helper, then delete memory/r-personal/
/// after its exit (on the store queue). Completion on main after deletion.
- (void)clearAllWithCompletion:(nullable dispatch_block_t)completion;

/// The version the selector should load (nil = factory weights).
- (nullable AFMPersonalVersionRequest *)activeVersionRequest;

/// Reads status.json (off main); completion on main with nil when absent.
- (void)readStatusWithCompletion:(void (^)(NSDictionary *_Nullable status))completion;

// MARK: Scheduler (design D6)
/// Creates the NSBackgroundActivityScheduler (identifier
/// <bundle id>.personal-finetune, 24 h, tolerance 6 h, repeats, background).
- (void)startScheduler;
- (void)stopScheduler;
@property (readonly, nonatomic, nullable) NSBackgroundActivityScheduler *scheduler;
/// The scheduler block body (also called by tests with a fake completion).
- (void)handleScheduledActivity:(void (^)(NSBackgroundActivityResult result))completion;

// MARK: Testing
/// Called on main after each finished run with its final result string
/// ("adopted", "not_significant", "failed:crash", "discarded:generation", ...).
@property (copy, nonatomic, nullable) void (^runDidFinishHandler)(NSString *result);
/// Waits (spinning the main run loop) until the store queue is idle and no
/// main-thread follow-up is pending.
- (void)waitUntilIdleForTesting;
@property (readonly, nonatomic) pid_t helperProcessIdentifierForTesting;
@property (readonly, nonatomic) NSUInteger spawnCountForTesting;

@end

NS_ASSUME_NONNULL_END
