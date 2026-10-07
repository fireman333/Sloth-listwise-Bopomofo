// Copyright (c) 2026 and onwards The McBopomofo Authors.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

#import "AFMPreciseController.h"
#import "AFMPersonalTraining.h"
#import "McBopomofo-Swift.h"

#import <malloc/malloc.h>
#import <os/log.h>

#import <atomic>
#import <chrono>
#import <cmath>
#import <memory>
#import <utility>

#include "Rescorer/qwen_rescorer.h"
#include "Rescorer/rescorer.h"

NSNotificationName const AFMPreciseControllerStateDidChangeNotification = @"AFMPreciseControllerStateDidChangeNotification";

static const NSUInteger kAFMPreciseMaxConsecutiveTimeouts = 3;

static NSString *AFMPreciseBundleIdentifier(void)
{
    NSString *identifier = NSBundle.mainBundle.bundleIdentifier;
    return identifier != nil ? identifier : @"io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo";
}

static os_log_t AFMPreciseLog(void)
{
    static os_log_t log;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        log = os_log_create(AFMPreciseBundleIdentifier().UTF8String, "Precise");
    });
    return log;
}

static const char *AFMPreciseStateName(AFMPreciseState state)
{
    switch (state) {
    case AFMPreciseStateOff:
        return "off";
    case AFMPreciseStateLoading:
        return "loading";
    case AFMPreciseStateReady:
        return "ready";
    case AFMPreciseStateNotFound:
        return "not found";
    case AFMPreciseStateFailed:
        return "failed";
    case AFMPreciseStateTripped:
        return "tripped";
    }
    return "unknown";
}

namespace {

// Test heavy model backed by an Objective-C block.
class AFMBlockRescorer : public McBopomofo::Rescorer {
public:
    explicit AFMBlockRescorer(AFMPreciseHeavyScoreBlock block)
        : block_([block copy])
    {
    }
    bool ready() const override { return true; }
    std::vector<double> score(const McBopomofo::RescoreRequest& request) override
    {
        @autoreleasepool {
            NSMutableArray<NSString *> *readings = [NSMutableArray arrayWithCapacity:request.readings.size()];
            for (const auto& r : request.readings) {
                [readings addObject:@(r.c_str())];
            }
            NSMutableArray<NSString *> *candidates = [NSMutableArray arrayWithCapacity:request.candidates.size()];
            for (const auto& c : request.candidates) {
                [candidates addObject:@(c.c_str())];
            }
            NSArray<NSNumber *> *scores = block_(readings, candidates);
            std::vector<double> out;
            for (NSNumber *n in scores) {
                out.push_back(n.doubleValue);
            }
            return out;
        }
    }

private:
    AFMPreciseHeavyScoreBlock block_;
};

} // namespace

// Loads the heavy model from the model directory (runs on the queue). Returns
// nullptr with *error set; *notFound = YES when the model file is missing.
typedef McBopomofo::Rescorer * (^AFMPreciseHeavyLoader)(NSString *modelDirectory, NSString **error, BOOL *notFound);

static McBopomofo::Rescorer *AFMLoadQwenRescorer(NSString *modelDirectory, int threads, NSString **error, BOOL *notFound)
{
    NSString *gguf = [modelDirectory stringByAppendingPathComponent:@"model.gguf"];
    if (![NSFileManager.defaultManager fileExistsAtPath:gguf]) {
        *error = @"heavy model file missing (model.gguf)";
        *notFound = YES;
        return nullptr;
    }
    auto *model = new McBopomofo::QwenRescorer(std::string(gguf.fileSystemRepresentation), 99, threads);
    if (!model->ready()) {
        delete model;
        *error = @"failed to load the heavy model (model.gguf)";
        return nullptr;
    }
    return model;
}

#pragma mark - AFMPreciseRequest

@interface AFMPreciseRequest ()
@property (readwrite, nonatomic) NSUInteger epoch;
@end

@implementation AFMPreciseRequest {
    std::vector<std::string> _readings;
    std::vector<std::string> _candidates;
    std::vector<double> _rScores;
}

@synthesize key = _key;
@synthesize composingText = _composingText;
@synthesize requester = _requester;
@synthesize epoch = _epoch;

- (instancetype)initWithReadings:(std::vector<std::string>)readings
                      candidates:(std::vector<std::string>)candidates
                         rScores:(std::vector<double>)rScores
                             key:(NSString *)key
                   composingText:(NSString *)composingText
                       requester:(id<AFMPreciseRequester>)requester
{
    self = [super init];
    if (self) {
        _readings = std::move(readings);
        _candidates = std::move(candidates);
        _rScores = std::move(rScores);
        _key = [key copy];
        _composingText = [composingText copy];
        _requester = requester;
    }
    return self;
}

- (instancetype)initForTestingWithReadings:(NSArray<NSString *> *)readings candidates:(NSArray<NSString *> *)candidates rScores:(NSArray<NSNumber *> *)rScores key:(NSString *)key composingText:(NSString *)composingText requester:(id<AFMPreciseRequester>)requester
{
    std::vector<std::string> r, c;
    std::vector<double> s;
    for (NSString *x in readings) {
        r.emplace_back(x.UTF8String);
    }
    for (NSString *x in candidates) {
        c.emplace_back(x.UTF8String);
    }
    for (NSNumber *x in rScores) {
        s.push_back(x.doubleValue);
    }
    return [self initWithReadings:std::move(r) candidates:std::move(c) rScores:std::move(s) key:key composingText:composingText requester:requester];
}

- (const std::vector<std::string> &)readings
{
    return _readings;
}

- (const std::vector<std::string> &)candidates
{
    return _candidates;
}

- (const std::vector<double> &)rScores
{
    return _rScores;
}

@end

#pragma mark - AFMPreciseController

static AFMPreciseController *gSharedPreciseController = nil;

// Production heavy-model settings (design D2 addendum).
static const NSInteger kAFMPreciseQwenThreads = 4;
static const qos_class_t kAFMPreciseQueueQoS = QOS_CLASS_UTILITY;
// A heavy inference does not start until this long after the last key event
// (design D2 addendum): it runs in typing pauses instead of competing with
// the next keystroke's R inference.
static const NSTimeInterval kAFMPreciseStartDelay = 0.04;

static double AFMNowSeconds()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

@interface AFMPreciseController ()
- (instancetype)_initWithModelDirectory:(nullable NSString *)modelDirectory
                                 loader:(AFMPreciseHeavyLoader)loader
                              loadDelay:(NSTimeInterval)loadDelay
                                    qos:(qos_class_t)qos NS_DESIGNATED_INITIALIZER;
@end

@implementation AFMPreciseController {
    AFMPreciseHeavyLoader _loader;
    NSTimeInterval _loadDelay;
    dispatch_queue_t _queue;
    // QoS of the queue and of every block run on it (design D2 addendum).
    qos_class_t _qos;
    // Queue-confined.
    std::unique_ptr<McBopomofo::Rescorer> _heavy;
    // Main thread.
    McBopomofo::PreciseConfig _config;
    AFMPreciseRequest *_running;
    AFMPreciseRequest *_pending;
    double _lastKeyEventTime;
    BOOL _startTimerScheduled;
    BOOL _tripped;
    double _loadedAlpha;
    // Mirror of _epoch for the queue, so stale loads / inferences are skipped.
    std::atomic<NSUInteger> _queueEpoch;
    // Personal-finetune D12 (main thread): factory calibration, the selected
    // personalized version and its parsed overlay.
    McBopomofo::PreciseConfig _baseConfig;
    AFMPersonalVersionRequest *_personalVersion;
    BOOL _overlayValid;
    std::vector<int> _overlayBuckets;
    std::vector<double> _overlayTemperatures;
    BOOL _overlayFailed;
    NSUInteger _overlayToken;
}

@synthesize state = _state;
@synthesize enabled = _enabled;
@synthesize epoch = _epoch;
@synthesize modelDirectory = _modelDirectory;
@synthesize lastLoadMilliseconds = _lastLoadMilliseconds;
@synthesize submittedCount = _submittedCount;
@synthesize scoredCount = _scoredCount;
@synthesize deliveredCount = _deliveredCount;
@synthesize timeoutStreak = _timeoutStreak;
@synthesize discardedLoadCount = _discardedLoadCount;
@synthesize lastHeavyQoSClass = _lastHeavyQoSClass;
@synthesize overlayUpdateCount = _overlayUpdateCount;
@synthesize startDelay = _startDelay;
@synthesize totalHeavyMilliseconds = _totalHeavyMilliseconds;
@synthesize maxHeavyMilliseconds = _maxHeavyMilliseconds;
@synthesize deferredStartCount = _deferredStartCount;

+ (AFMPreciseController *)sharedController
{
    if (gSharedPreciseController == nil) {
        gSharedPreciseController = [[AFMPreciseController alloc] initWithModelDirectory:UserPhraseLocationHelper.qwenModelDirectory];
    }
    return gSharedPreciseController;
}

+ (void)setSharedController:(AFMPreciseController *)controller
{
    gSharedPreciseController = controller;
}

+ (void)resetForTesting
{
    if (gSharedPreciseController != nil) {
        gSharedPreciseController.enabled = NO;
    }
    NSString *missing = [NSTemporaryDirectory() stringByAppendingPathComponent:[@"no-such-qwen-" stringByAppendingString:NSUUID.UUID.UUIDString]];
    self.sharedController = [[AFMPreciseController alloc] initWithModelDirectory:missing];
}

- (instancetype)_initWithModelDirectory:(NSString *)modelDirectory loader:(AFMPreciseHeavyLoader)loader loadDelay:(NSTimeInterval)loadDelay qos:(qos_class_t)qos
{
    self = [super init];
    if (self) {
        _modelDirectory = [modelDirectory copy];
        _loader = [loader copy];
        _loadDelay = loadDelay;
        _state = AFMPreciseStateOff;
        _queueEpoch = 0;
        NSString *label = [AFMPreciseBundleIdentifier() stringByAppendingString:@".Precise"];
        // Low QoS so the heavy model (and the llama/ggml worker threads it
        // spawns, which inherit the creating thread's QoS) yields CPU to the
        // main thread's keystroke handling.
        _qos = qos;
        _startDelay = kAFMPreciseStartDelay;
        _lastKeyEventTime = -INFINITY;
        _queue = dispatch_queue_create(label.UTF8String,
                                       dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, qos, 0));
    }
    return self;
}

- (instancetype)initWithModelDirectory:(NSString *)modelDirectory
{
    return [self initWithModelDirectory:modelDirectory qwenThreadsForTesting:kAFMPreciseQwenThreads queueQoSForTesting:kAFMPreciseQueueQoS];
}

- (instancetype)initWithModelDirectory:(NSString *)modelDirectory qwenThreadsForTesting:(NSInteger)threads
{
    return [self initWithModelDirectory:modelDirectory qwenThreadsForTesting:threads queueQoSForTesting:kAFMPreciseQueueQoS];
}

- (instancetype)initWithModelDirectory:(NSString *)modelDirectory qwenThreadsForTesting:(NSInteger)threads queueQoSForTesting:(qos_class_t)qos
{
    int n = static_cast<int>(threads);
    return [self _initWithModelDirectory:modelDirectory
                                  loader:^McBopomofo::Rescorer *(NSString *directory, NSString **error, BOOL *notFound) {
                                      return AFMLoadQwenRescorer(directory, n, error, notFound);
                                  }
                               loadDelay:0
                                     qos:qos];
}

- (instancetype)initWithModelDirectory:(NSString *)modelDirectory heavyScoreBlockForTesting:(AFMPreciseHeavyScoreBlock)scoreBlock loadDelay:(NSTimeInterval)loadDelay
{
    AFMPreciseHeavyScoreBlock block = [scoreBlock copy];
    return [self _initWithModelDirectory:modelDirectory
                                  loader:^McBopomofo::Rescorer *(NSString *directory, NSString **error, BOOL *notFound) {
                                      return new AFMBlockRescorer(block);
                                  }
                               loadDelay:loadDelay
                                     qos:kAFMPreciseQueueQoS];
}

- (BOOL)isReady
{
    return _state == AFMPreciseStateReady;
}

- (BOOL)isHeavyRunning
{
    return _running != nil;
}

- (BOOL)isIdle
{
    return _running == nil && _pending == nil;
}

- (const McBopomofo::PreciseConfig &)config
{
    return _config;
}

- (void)_setState:(AFMPreciseState)state
{
    if (_state == state) {
        return;
    }
    _state = state;
    os_log_info(AFMPreciseLog(), "precise mode: %{public}s", AFMPreciseStateName(state));
    [NSNotificationCenter.defaultCenter postNotificationName:AFMPreciseControllerStateDidChangeNotification object:self];
}

- (void)_bumpEpoch
{
    ++_epoch;
    _queueEpoch.store(_epoch);
    _pending = nil;
}

#pragma mark - Enable / disable

- (void)setEnabled:(BOOL)enabled
{
    if (enabled == _enabled) {
        return;
    }
    _enabled = enabled;
    if (enabled) {
        [self _startLoad];
    } else {
        [self _unload];
    }
}

- (void)syncWithPreferences
{
    BOOL wanted = Preferences.preciseModeEnabled;
    if (wanted != _enabled) {
        self.enabled = wanted;
        return;
    }
    if (_enabled && !_tripped && Preferences.neuralWalkAlpha != _loadedAlpha) {
        [self _startLoad];
    }
}

- (void)_startLoad
{
    [self _bumpEpoch];
    if (_tripped) {
        return;
    }
    [self _setState:AFMPreciseStateLoading];
    double alpha = Preferences.neuralWalkAlpha;
    _loadedAlpha = alpha;
    NSUInteger epoch = _epoch;
    NSString *directory = _modelDirectory;
    AFMPreciseHeavyLoader loader = _loader;
    NSTimeInterval loadDelay = _loadDelay;
    dispatch_async(_queue, ^{
        if (self->_queueEpoch.load() != epoch) {
            return;  // Superseded before it started.
        }
        // Release the previous model first, so two never coexist.
        self->_heavy.reset();
        auto t0 = std::chrono::steady_clock::now();
        AFMPreciseState outcome = AFMPreciseStateReady;
        NSString *error = nil;
        McBopomofo::PreciseConfig config;
        BOOL isDirectory = NO;
        NSString *confPath = [directory stringByAppendingPathComponent:@"precise.conf"];
        if (directory == nil || ![NSFileManager.defaultManager fileExistsAtPath:directory isDirectory:&isDirectory] || !isDirectory) {
            outcome = AFMPreciseStateNotFound;
            error = @"model directory missing";
        } else if (![NSFileManager.defaultManager fileExistsAtPath:confPath]) {
            outcome = AFMPreciseStateNotFound;
            error = @"calibration file missing (precise.conf)";
        } else {
            std::string message;
            if (!McBopomofo::PreciseConfig::LoadFile(confPath.fileSystemRepresentation, &config, &message)) {
                outcome = AFMPreciseStateFailed;
                error = [NSString stringWithFormat:@"invalid calibration file: %s", message.c_str()];
            } else if (std::fabs(config.alpha - alpha) > 1e-9) {
                outcome = AFMPreciseStateFailed;
                error = [NSString stringWithFormat:@"calibration alpha %g differs from the R-walk alpha %g", config.alpha, alpha];
            }
        }
        if (outcome == AFMPreciseStateReady) {
            if (loadDelay > 0) {
                [NSThread sleepForTimeInterval:loadDelay];
            }
            BOOL notFound = NO;
            McBopomofo::Rescorer *heavy = nullptr;
            @try {
                heavy = loader(directory, &error, &notFound);
            } @catch (NSException *exception) {
                heavy = nullptr;
                error = exception.reason;
            }
            if (heavy == nullptr) {
                outcome = notFound ? AFMPreciseStateNotFound : AFMPreciseStateFailed;
            } else if (self->_queueEpoch.load() != epoch) {
                delete heavy;  // Disabled while loading; never keep it.
                heavy = nullptr;
            } else {
                self->_heavy.reset(heavy);
            }
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        dispatch_async(dispatch_get_main_queue(), ^{
            [self _finishLoadWithEpoch:epoch outcome:outcome config:config error:error milliseconds:ms];
        });
    });
}

- (void)_finishLoadWithEpoch:(NSUInteger)epoch outcome:(AFMPreciseState)outcome config:(McBopomofo::PreciseConfig)config error:(NSString *)error milliseconds:(double)ms
{
    if (epoch != _epoch || _state != AFMPreciseStateLoading) {
        ++_discardedLoadCount;
        return;  // From an earlier enable / disable cycle.
    }
    _lastLoadMilliseconds = ms;
    if (outcome != AFMPreciseStateReady) {
        os_log_error(AFMPreciseLog(), "precise mode unavailable, using R-walk: %{public}@", error ?: @"unknown error");
        [self _setState:outcome];
        return;
    }
    _baseConfig = config;
    _config = config;
    _timeoutStreak = 0;
    os_log_info(AFMPreciseLog(), "precise mode model ready in %.0f ms", ms);
    [self _setState:AFMPreciseStateReady];
    [self _applyPersonalOverlay];
}

#pragma mark - Personalized R overlay (personal-finetune D12)

// "key = value" lines: run, bucket_upper, temperatures (# comments).
static BOOL AFMParsePreciseOverlay(NSString *path, NSString **run, std::vector<int> *buckets, std::vector<double> *temperatures)
{
    NSString *text = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    if (text == nil) {
        return NO;
    }
    BOOL hasBuckets = NO, hasTemperatures = NO;
    for (NSString *raw in [text componentsSeparatedByString:@"\n"]) {
        NSString *line = [raw stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
        if (line.length == 0 || [line hasPrefix:@"#"]) {
            continue;
        }
        NSRange eq = [line rangeOfString:@"="];
        if (eq.location == NSNotFound) {
            return NO;
        }
        NSString *key = [[line substringToIndex:eq.location] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
        NSString *value = [[line substringFromIndex:eq.location + 1] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
        NSMutableArray<NSString *> *parts = [NSMutableArray array];
        for (NSString *p in [value componentsSeparatedByString:@" "]) {
            if (p.length > 0) {
                [parts addObject:p];
            }
        }
        if ([key isEqualToString:@"run"]) {
            *run = value;
        } else if ([key isEqualToString:@"bucket_upper"]) {
            hasBuckets = YES;
            buckets->clear();
            for (NSString *p in parts) {
                NSScanner *scanner = [NSScanner scannerWithString:p];
                int v = 0;
                if (![scanner scanInt:&v] || !scanner.isAtEnd) {
                    return NO;
                }
                buckets->push_back(v);
            }
        } else if ([key isEqualToString:@"temperatures"]) {
            hasTemperatures = YES;
            temperatures->clear();
            for (NSString *p in parts) {
                NSScanner *scanner = [NSScanner scannerWithString:p];
                double v = 0;
                if (![scanner scanDouble:&v] || !scanner.isAtEnd || !std::isfinite(v) || v <= 0) {
                    return NO;
                }
                temperatures->push_back(v);
            }
        } else {
            return NO;
        }
    }
    return *run != nil && hasBuckets && hasTemperatures && temperatures->size() == buckets->size() + 1;
}

- (void)setPersonalVersion:(AFMPersonalVersionRequest *)request
{
    NSUInteger token = ++_overlayToken;
    _personalVersion = request;
    if (request == nil) {
        _overlayValid = NO;
        ++_overlayUpdateCount;
        [self _applyPersonalOverlay];
        return;
    }
    NSString *path = [request.directory stringByAppendingPathComponent:@"precise-r.conf"];
    NSString *expectedRun = [request.runIdentifier copy];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        NSString *run = nil;
        std::vector<int> buckets;
        std::vector<double> temperatures;
        BOOL ok = AFMParsePreciseOverlay(path, &run, &buckets, &temperatures) && [run isEqualToString:expectedRun];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (token != self->_overlayToken) {
                return;  // Another version was selected meanwhile.
            }
            self->_overlayValid = ok;
            self->_overlayBuckets = buckets;
            self->_overlayTemperatures = temperatures;
            ++self->_overlayUpdateCount;
            [self _applyPersonalOverlay];
        });
    });
}

- (NSArray<NSNumber *> *)configTemperaturesForTesting
{
    NSMutableArray<NSNumber *> *result = [NSMutableArray array];
    for (double t : _config.temperatures) {
        [result addObject:@(t)];
    }
    return result;
}

- (BOOL)isPersonalOverlayApplied
{
    return _personalVersion != nil && _overlayValid && !_overlayFailed && _state == AFMPreciseStateReady;
}

/// Recomputes _config from the factory calibration and the overlay; an
/// unusable overlay for a personalized R makes precise mode failed.
- (void)_applyPersonalOverlay
{
    BOOL readyOrOverlayFailed = _state == AFMPreciseStateReady || (_state == AFMPreciseStateFailed && _overlayFailed);
    if (!readyOrOverlayFailed) {
        return;
    }
    if (_personalVersion == nil) {
        _config = _baseConfig;
        if (_overlayFailed) {
            _overlayFailed = NO;
            [self _setState:AFMPreciseStateReady];
        }
        return;
    }
    if (_overlayValid && _overlayBuckets == _baseConfig.bucketUpper) {
        McBopomofo::PreciseConfig config = _baseConfig;
        config.temperatures = _overlayTemperatures;
        _config = config;
        if (_overlayFailed) {
            _overlayFailed = NO;
            [self _setState:AFMPreciseStateReady];
        }
        return;
    }
    if (_state == AFMPreciseStateReady) {
        _overlayFailed = YES;
        [self _bumpEpoch];
        os_log_error(AFMPreciseLog(), "precise mode unavailable for personalized R version %ld: calibration missing or not for this version", (long)_personalVersion.version);
        [self _setState:AFMPreciseStateFailed];
    }
}

- (void)_unload
{
    [self _bumpEpoch];
    if (!_tripped) {
        [self _setState:AFMPreciseStateOff];
    }
    [self _releaseModelOnQueue];
}

- (void)_releaseModelOnQueue
{
    NSUInteger epoch = _epoch;
    dispatch_async(_queue, ^{
        // Runs after any inference already queued; a later load replaces it.
        if (self->_queueEpoch.load() == epoch && self->_heavy != nullptr) {
            self->_heavy.reset();
            malloc_zone_pressure_relief(NULL, 0);
        }
    });
}

#pragma mark - Requests

- (void)submitRequest:(AFMPreciseRequest *)request
{
    if (_state != AFMPreciseStateReady || request == nil) {
        return;
    }
    request.epoch = _epoch;
    ++_submittedCount;
    // Latest wins: any older queued request is dropped.
    _pending = request;
    [self _runPendingIfPossible];
}

- (void)noteKeyEvent
{
    _lastKeyEventTime = AFMNowSeconds();
}

// Seconds until a heavy inference may start (0 = now).
- (double)_startWait
{
    double wait = _lastKeyEventTime + _startDelay - AFMNowSeconds();
    return wait > 0 ? wait : 0;
}

- (void)_run:(AFMPreciseRequest *)request
{
    _running = request;
    NSUInteger epoch = request.epoch;
    McBopomofo::PreciseConfig config = _config;
    // ENFORCE: run at the queue's low QoS, not the submitting main thread's
    // user-interactive QoS.
    dispatch_block_t work = dispatch_block_create_with_qos_class(DISPATCH_BLOCK_ENFORCE_QOS_CLASS, _qos, 0, ^{
        McBopomofo::PreciseDecision decision;
        BOOL ran = NO;
        qos_class_t observed = qos_class_self();
        if (self->_queueEpoch.load() == epoch && self->_heavy != nullptr) {
            decision = McBopomofo::PreciseSelect(request.readings, request.candidates, request.rScores, self->_heavy.get(), config);
            ran = decision.heavyCalled;
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            self->_lastHeavyQoSClass = observed;
            if (ran) {
                self->_totalHeavyMilliseconds += decision.heavyMs;
                self->_maxHeavyMilliseconds = std::max(self->_maxHeavyMilliseconds, decision.heavyMs);
            }
            [self _finishRequest:request decision:decision ran:ran];
        });
    });
    dispatch_async(_queue, work);
}

- (void)_finishRequest:(AFMPreciseRequest *)request decision:(const McBopomofo::PreciseDecision &)decision ran:(BOOL)ran
{
    if (_running == request) {
        _running = nil;
    }
    if (ran) {
        ++_scoredCount;
    }
    if (ran && request.epoch == _epoch && _state == AFMPreciseStateReady) {
        if (decision.heavyTimedOut) {
            ++_timeoutStreak;
            os_log_info(AFMPreciseLog(), "precise mode: heavy inference timed out (%.0f ms, %lu in a row)", decision.heavyMs, (unsigned long)_timeoutStreak);
            if (_timeoutStreak >= kAFMPreciseMaxConsecutiveTimeouts) {
                [self _trip];
            }
        } else {
            _timeoutStreak = 0;
            if (!decision.heavyOk) {
                os_log_debug(AFMPreciseLog(), "precise mode: heavy model gave no usable scores (%.0f ms)", decision.heavyMs);
            } else if (decision.chosen != 0) {
                id<AFMPreciseRequester> requester = request.requester;
                if (requester != nil) {
                    ++_deliveredCount;
                    [requester preciseRequestDidFinish:request chosenIndex:decision.chosen];
                }
            }
        }
    }
    [self _runPendingIfPossible];
}

- (void)_runPendingIfPossible
{
    if (_running != nil || _pending == nil) {
        return;
    }
    double wait = [self _startWait];
    if (wait > 0) {
        // Too soon after a key event: retry when the delay has passed (a
        // later key event pushes it further; a newer request replaces this one).
        if (!_startTimerScheduled) {
            _startTimerScheduled = YES;
            ++_deferredStartCount;
            __weak AFMPreciseController *weakSelf = self;
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(wait * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
                AFMPreciseController *strongSelf = weakSelf;
                if (strongSelf == nil) {
                    return;
                }
                strongSelf->_startTimerScheduled = NO;
                [strongSelf _runPendingIfPossible];
            });
        }
        return;
    }
    AFMPreciseRequest *next = _pending;
    _pending = nil;
    if (_state != AFMPreciseStateReady || next.epoch != _epoch) {
        return;
    }
    [self _run:next];
}

- (void)_trip
{
    _tripped = YES;
    [self _bumpEpoch];
    os_log_error(AFMPreciseLog(), "precise mode disabled for this process: %lu consecutive heavy inferences over the timeout", (unsigned long)kAFMPreciseMaxConsecutiveTimeouts);
    [self _setState:AFMPreciseStateTripped];
    [self _releaseModelOnQueue];
}

@end
