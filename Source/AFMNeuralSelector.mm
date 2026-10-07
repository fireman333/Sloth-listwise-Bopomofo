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

#import "AFMNeuralSelector.h"
#import "AFMPersonalTraining.h"
#import "AFMPreciseController.h"
#import "McBopomofo-Swift.h"

#import <malloc/malloc.h>
#import <os/log.h>

#import <chrono>
#import <cmath>
#import <string>
#import <vector>

// ORT_API_MANUAL_INIT is defined for the whole target (build settings), so the
// onnxruntime C++ headers never call OrtGetApiBase() on their own.
#include <onnxruntime_cxx_api.h>

#include "Rescorer/neural_walk.h"
#include "Rescorer/sloth_rescorer.h"

// libonnxruntime is weak-linked: when the dylib is missing from
// Contents/Frameworks, dyld still launches the app and this symbol is null.
// Redeclare it weak so the compiler keeps the null check.
extern "C" __attribute__((weak_import)) const OrtApiBase *ORT_API_CALL OrtGetApiBase(void) NO_EXCEPTION;

const NSUInteger kAFMNeuralMaxSyllables = 40;
NSNotificationName const AFMNeuralSelectorStateDidChangeNotification = @"AFMNeuralSelectorStateDidChangeNotification";

// The app's bundle identifier, used for the log subsystem and queue labels.
static NSString *AFMBundleIdentifier(void)
{
    NSString *identifier = NSBundle.mainBundle.bundleIdentifier;
    return identifier != nil ? identifier : @"io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo";
}

static os_log_t AFMNeuralLog(void)
{
    static os_log_t log;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        log = os_log_create(AFMBundleIdentifier().UTF8String, "RWalk");
    });
    return log;
}

// Identifies the loaded R model for the sentence pool (personal-memory D4):
// "<model file name>:<size>:<mtime>", or "test" for a test scorer. Runs on
// the load queue.
static NSString *AFMModelIdentifier(NSString *modelDirectory)
{
    if (modelDirectory == nil) {
        return @"test";
    }
    NSString *path = [modelDirectory stringByAppendingPathComponent:@"onnx/model_quantized.onnx"];
    NSDictionary *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:path error:nil];
    if (attributes == nil) {
        return @"r-walk";
    }
    return [NSString stringWithFormat:@"%@:%llu:%lld", path.lastPathComponent, attributes.fileSize,
                     (long long)attributes.fileModificationDate.timeIntervalSince1970];
}

// A loaded scorer and the model it wraps (nullptr for test scorers). Freed
// on the load queue when replaced (personal-finetune D11); the scorer in use
// at process exit is intentionally never freed (no onnxruntime teardown).
struct AFMLoadedScorer {
    McBopomofo::NeuralScorer *scorer = nullptr;
    McBopomofo::SlothRescorer *model = nullptr;
};

static void AFMFreeLoadedScorer(AFMLoadedScorer loaded)
{
    delete loaded.scorer;
    delete loaded.model;
}

/// Outcome of one load (load queue -> main).
@interface AFMNeuralLoadResult : NSObject
@property (nonatomic) AFMLoadedScorer loaded;
@property (copy, nonatomic, nullable) NSString *error;
@property (nonatomic) BOOL notFound;
/// Personalized weights requested: applied, or why not.
@property (nonatomic) BOOL personalApplied;
@property (nonatomic) BOOL personalMismatch;
@property (copy, nonatomic, nullable) NSString *personalError;
@end

@implementation AFMNeuralLoadResult
@synthesize loaded = _loaded;
@synthesize error = _error;
@synthesize notFound = _notFound;
@synthesize personalApplied = _personalApplied;
@synthesize personalMismatch = _personalMismatch;
@synthesize personalError = _personalError;
@end

// The configuration a personalized load is checked against (main thread
// snapshot; the factory SHA / pack version are read on the load queue).
struct AFMLoadConfig {
    double alpha = 0.5;
    bool acceptYan = false;
};

// Production loader. Runs on the load queue. Factory weights, or the
// requested personalized version when its eval_config matches the current
// configuration (design D11); a personalized problem falls back to the
// factory weights with personalError set.
static AFMNeuralLoadResult *AFMLoadSlothScorer(NSString *modelDirectory, AFMPersonalVersionRequest *request, AFMLoadConfig config)
{
    AFMNeuralLoadResult *result = [[AFMNeuralLoadResult alloc] init];
    if (modelDirectory == nil) {
        result.error = @"no model directory configured";
        result.notFound = YES;
        return result;
    }
    if (&OrtGetApiBase == nullptr) {
        result.error = @"onnxruntime is not available (libonnxruntime.1.dylib missing)";
        return result;
    }
    BOOL isDirectory = NO;
    if (![[NSFileManager defaultManager] fileExistsAtPath:modelDirectory isDirectory:&isDirectory] || !isDirectory) {
        result.error = [NSString stringWithFormat:@"model directory missing: %@", modelDirectory];
        result.notFound = YES;
        return result;
    }
    const OrtApiBase *apiBase = OrtGetApiBase();
    const OrtApi *api = apiBase != nullptr ? apiBase->GetApi(ORT_API_VERSION) : nullptr;
    if (api == nullptr) {
        result.error = [NSString stringWithFormat:@"onnxruntime does not provide API version %d", ORT_API_VERSION];
        return result;
    }
    Ort::InitApi(api);

    McBopomofo::SlothRescorer::Options options;
    if (request != nil) {
        options.overridesPath = std::string([request.directory stringByAppendingPathComponent:@"weights.bin"].fileSystemRepresentation);
    }
    McBopomofo::SlothRescorer *model = nullptr;
    try {
        model = new McBopomofo::SlothRescorer(std::string(modelDirectory.fileSystemRepresentation), options);
    } catch (...) {
        model = nullptr;
    }
    if (model == nullptr || !model->ready()) {
        delete model;
        result.error = [NSString stringWithFormat:@"failed to load the R model from %@", modelDirectory];
        return result;
    }
    if (!options.overridesPath.empty()) {
        if (model->personalized()) {
            result.personalApplied = YES;
        } else {
            result.personalError = @(model->personalizationError().c_str());
        }
    }
    AFMLoadedScorer loaded;
    loaded.model = model;
    loaded.scorer = new McBopomofo::NeuralScorer(McBopomofo::MakeNeuralScorer(model));
    result.loaded = loaded;
    return result;
}

// Wraps a test score block as a NeuralScorer (called on the main thread).
static McBopomofo::NeuralScorer *AFMMakeBlockScorer(AFMNeuralScoreBlock block)
{
    return new McBopomofo::NeuralScorer([block](const std::vector<std::string> &readings, const std::vector<std::vector<std::string>> *extra) {
        McBopomofo::SlothRescorer::CharLogProbTable table;
        NSMutableArray<NSString *> *readingArray = [NSMutableArray arrayWithCapacity:readings.size()];
        for (const auto &r : readings) {
            [readingArray addObject:@(r.c_str())];
        }
        NSMutableArray<NSArray<NSString *> *> *charArray = [NSMutableArray array];
        if (extra != nullptr) {
            for (const auto &position : *extra) {
                NSMutableArray<NSString *> *chars = [NSMutableArray arrayWithCapacity:position.size()];
                for (const auto &c : position) {
                    [chars addObject:@(c.c_str())];
                }
                [charArray addObject:chars];
            }
        }
        NSArray<NSDictionary<NSString *, NSNumber *> *> *result = block(readingArray, charArray);
        for (NSDictionary<NSString *, NSNumber *> *position in result) {
            std::unordered_map<std::string, double> map;
            for (NSString *key in position) {
                map[key.UTF8String] = position[key].doubleValue;
            }
            table.push_back(std::move(map));
        }
        return table;
    });
}

typedef AFMNeuralLoadResult * (^AFMNeuralLoader)(AFMPersonalVersionRequest *_Nullable request, AFMLoadConfig config);
typedef NSDictionary * (^AFMEvalConfigProvider)(double alpha, BOOL acceptYan);

static AFMNeuralSelector *gSharedSelector = nil;

@interface AFMNeuralSelector ()
- (instancetype)_initWithModelDirectory:(nullable NSString *)modelDirectory
                                 loader:(AFMNeuralLoader)loader NS_DESIGNATED_INITIALIZER;
@end

@implementation AFMNeuralSelector {
    AFMNeuralLoader _loader;
    dispatch_queue_t _loadQueue;
    // The scorer in use (main thread only). Replaced scorers are freed on the
    // load queue; the last one is never freed (no onnxruntime teardown).
    AFMLoadedScorer _loaded;
    McBopomofo::NeuralScorer *_scorer;
    McBopomofo::NeuralScoreCache _cache;
    McBopomofo::NeuralCircuitBreaker _breaker;
    // A cache-miss measurement kept for the end-of-key-event hook (design D1).
    BOOL _hasPendingBreakerMs;
    double _pendingBreakerMs;
    // Personalization (design D11): key of the weights in use / of the newest
    // load started (a completion with another key is discarded).
    NSString *_loadedKey;
    NSString *_pendingKey;
}

@synthesize state = _state;
@synthesize enabled = _enabled;
@synthesize alpha = _alpha;
@synthesize modelDirectory = _modelDirectory;
@synthesize modelIdentifier = _modelIdentifier;
@synthesize lastLoadMilliseconds = _lastLoadMilliseconds;
@synthesize forwardCount = _forwardCount;
@synthesize neuralWalkCount = _neuralWalkCount;
@synthesize breakerRecordCount = _breakerRecordCount;
@synthesize precisePayloadCount = _precisePayloadCount;
@synthesize circuitBreakerDisabledForTesting = _circuitBreakerDisabledForTesting;
@synthesize personalStatus = _personalStatus;
@synthesize loadedPersonalVersion = _loadedPersonalVersion;
@synthesize personalError = _personalError;
@synthesize loadCount = _loadCount;
@synthesize discardedLoadCount = _discardedLoadCount;
@synthesize freedScorerCount = _freedScorerCount;
@synthesize personalVersionProvider = _personalVersionProvider;
@synthesize currentEvalConfigProvider = _currentEvalConfigProvider;

+ (AFMNeuralSelector *)sharedSelector
{
    if (gSharedSelector == nil) {
        gSharedSelector = [[AFMNeuralSelector alloc] initWithModelDirectory:UserPhraseLocationHelper.rWalkModelDirectory];
    }
    return gSharedSelector;
}

+ (void)setSharedSelector:(AFMNeuralSelector *)selector
{
    gSharedSelector = selector;
}

+ (void)resetForTesting
{
    self.sharedSelector = [[AFMNeuralSelector alloc] initWithModelDirectory:nil];
}

- (instancetype)_initWithModelDirectory:(NSString *)modelDirectory loader:(AFMNeuralLoader)loader
{
    self = [super init];
    if (self) {
        _modelDirectory = [modelDirectory copy];
        _loader = [loader copy];
        _enabled = YES;
        _alpha = 0.5;
        _state = AFMNeuralSelectorStateDisabled;
        NSString *label = [AFMBundleIdentifier() stringByAppendingString:@".RWalkLoad"];
        _loadQueue = dispatch_queue_create(label.UTF8String, DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

- (instancetype)initWithModelDirectory:(NSString *)modelDirectory
{
    NSString *directory = [modelDirectory copy];
    return [self _initWithModelDirectory:directory loader:^AFMNeuralLoadResult *(AFMPersonalVersionRequest *request, AFMLoadConfig config) {
        return AFMLoadSlothScorer(directory, request, config);
    }];
}

- (instancetype)initWithScoreBlockForTesting:(AFMNeuralScoreBlock)scoreBlock loadDelay:(NSTimeInterval)loadDelay
{
    return [self initWithScoreBlockForTesting:scoreBlock personalBlockProvider:nil loadDelay:loadDelay];
}

- (instancetype)initWithScoreBlockForTesting:(AFMNeuralScoreBlock)scoreBlock personalBlockProvider:(AFMPersonalScoreBlockProvider)personalBlockProvider loadDelay:(NSTimeInterval)loadDelay
{
    AFMNeuralScoreBlock block = [scoreBlock copy];
    AFMPersonalScoreBlockProvider provider = [personalBlockProvider copy];
    return [self _initWithModelDirectory:nil loader:^AFMNeuralLoadResult *(AFMPersonalVersionRequest *request, AFMLoadConfig config) {
        if (loadDelay > 0) {
            [NSThread sleepForTimeInterval:loadDelay];
        }
        AFMNeuralLoadResult *result = [[AFMNeuralLoadResult alloc] init];
        AFMNeuralScoreBlock use = block;
        if (request != nil) {
            NSString *error = nil;
            AFMNeuralScoreBlock personal = nil;
            if (provider != nil) {
                personal = provider(request, &error);
            }
            if (personal != nil) {
                use = personal;
                result.personalApplied = YES;
            } else {
                result.personalError = (error != nil ? error : @"no personalized scorer");
            }
        }
        AFMLoadedScorer loaded;
        loaded.scorer = AFMMakeBlockScorer(use);
        result.loaded = loaded;
        return result;
    }];
}

#pragma mark Loading

- (AFMPersonalVersionRequest *)_desiredRequest
{
    if (_personalVersionProvider != nil) {
        return _personalVersionProvider();
    }
    return AFMPersonalTrainingController.sharedController.activeVersionRequest;
}

// Identifies what a load produces: "f" (factory) or the version, run,
// generation and the configuration it is checked against.
static NSString *AFMLoadKey(AFMPersonalVersionRequest *request, AFMLoadConfig config)
{
    if (request == nil) {
        return @"f";
    }
    return [NSString stringWithFormat:@"v%ld|%@|%llu|%.17g|%d", (long)request.version, request.runIdentifier, request.generation, config.alpha, config.acceptYan ? 1 : 0];
}

- (AFMLoadConfig)_currentLoadConfig
{
    AFMLoadConfig config;
    config.alpha = _alpha;
    config.acceptYan = Preferences.acceptYanReadingForAi;
    return config;
}

- (void)syncWithPreferences
{
    _enabled = Preferences.neuralWalkEnabled;
    _alpha = Preferences.neuralWalkAlpha;
    [self startLoadingIfNeeded];
    [self _reloadIfNeeded];
}

- (void)startLoadingIfNeeded
{
    if (!_enabled || _state != AFMNeuralSelectorStateDisabled) {
        return;
    }
    [self _setState:AFMNeuralSelectorStateLoading];
    AFMLoadConfig config = [self _currentLoadConfig];
    [self _startLoad:[self _desiredRequest] config:config];
}

/// Activation: a different selected version (or configuration) is loaded in
/// the background while the current scorer stays in use (design D11).
- (void)_reloadIfNeeded
{
    if (_state != AFMNeuralSelectorStateReady) {
        return;
    }
    AFMLoadConfig config = [self _currentLoadConfig];
    AFMPersonalVersionRequest *request = [self _desiredRequest];
    NSString *key = AFMLoadKey(request, config);
    if ([key isEqualToString:(_pendingKey != nil ? _pendingKey : _loadedKey)]) {
        return;
    }
    [self _startLoad:request config:config];
}

- (void)requestFactoryWeights
{
    _personalVersionProvider = nil;
    AFMLoadConfig config = [self _currentLoadConfig];
    if (_state == AFMNeuralSelectorStateLoading) {
        // The load in flight may be personalized: discard it and load the
        // factory weights after it (serial queue).
        if (![_pendingKey isEqualToString:@"f"]) {
            [self _startLoad:nil config:config];
        }
        return;
    }
    if (_state != AFMNeuralSelectorStateReady) {
        return;
    }
    if ([_loadedKey isEqualToString:@"f"] && _pendingKey == nil) {
        return;
    }
    if (![_pendingKey isEqualToString:@"f"]) {
        [self _startLoad:nil config:config];
    }
}

- (BOOL)isReloading
{
    return _pendingKey != nil && _state == AFMNeuralSelectorStateReady;
}

- (void)_startLoad:(AFMPersonalVersionRequest *)request config:(AFMLoadConfig)config
{
    NSString *key = AFMLoadKey(request, config);
    _pendingKey = key;
    ++_loadCount;
    if (request != nil && _state == AFMNeuralSelectorStateReady) {
        _personalStatus = AFMPersonalModelStatusLoading;
    }
    AFMNeuralLoader loader = _loader;
    NSString *directory = _modelDirectory;
    AFMEvalConfigProvider configProvider = _currentEvalConfigProvider;
    dispatch_async(_loadQueue, ^{
        auto t0 = std::chrono::steady_clock::now();
        // Configuration binding (design D11): a version evaluated with another
        // factory model / alpha / reading policy / pack is not loaded.
        AFMPersonalVersionRequest *effective = request;
        BOOL mismatch = NO;
        if (request != nil) {
            NSDictionary *current = configProvider != nil ? configProvider(config.alpha, config.acceptYan) : AFMCurrentEvalConfig(AFMFactoryModelSHA256(directory), AFMPersonalPackVersion(directory), config.alpha, config.acceptYan);
            if (!AFMEvalConfigMatches(request.evalConfig, current)) {
                mismatch = YES;
                effective = nil;
            }
        }
        AFMNeuralLoadResult *result = nil;
        @try {
            result = loader(effective, config);
        } @catch (NSException *exception) {
            result = [[AFMNeuralLoadResult alloc] init];
            result.error = exception.reason;
        }
        if (mismatch) {
            result.personalMismatch = YES;
            result.personalError = @"version evaluated with another configuration";
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        NSString *identifier = nil;
        if (result.loaded.scorer != nullptr) {
            // Hand back the transient memory onnxruntime used while loading.
            malloc_zone_pressure_relief(NULL, 0);
            identifier = AFMModelIdentifier(directory);
            if (result.personalApplied) {
                identifier = [identifier stringByAppendingFormat:@"+p%ld", (long)request.version];
            }
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            [self _finishLoad:result key:key request:request identifier:identifier milliseconds:ms];
        });
    });
}

- (void)_freeOnLoadQueue:(AFMLoadedScorer)loaded
{
    if (loaded.scorer == nullptr && loaded.model == nullptr) {
        return;
    }
    ++_freedScorerCount;
    dispatch_async(_loadQueue, ^{
        AFMFreeLoadedScorer(loaded);
    });
}

- (void)_finishLoad:(AFMNeuralLoadResult *)result key:(NSString *)key request:(AFMPersonalVersionRequest *)request identifier:(NSString *)identifier milliseconds:(double)ms
{
    _lastLoadMilliseconds = ms;
    BOOL stale = ![key isEqualToString:_pendingKey];
    if (!stale && request != nil && request.generation != AFMPersonalTrainingController.sharedController.generation && _personalVersionProvider == nil) {
        stale = YES;  // Requested before a clear (design D13).
    }
    if (stale || (_state != AFMNeuralSelectorStateLoading && _state != AFMNeuralSelectorStateReady)) {
        // Superseded (e.g. a clear asked for the factory weights): never
        // published, freed off main.
        ++_discardedLoadCount;
        [self _freeOnLoadQueue:result.loaded];
        if (!stale) {
            _pendingKey = nil;
        }
        return;
    }
    _pendingKey = nil;
    if (result.loaded.scorer == nullptr) {
        if (_state == AFMNeuralSelectorStateLoading) {
            [self _setState:result.notFound ? AFMNeuralSelectorStateNotFound : AFMNeuralSelectorStateFailed];
        } else {
            // A reload failed: stop using R rather than keep weights that
            // were deselected.
            AFMLoadedScorer old = _loaded;
            _loaded = AFMLoadedScorer();
            _scorer = nullptr;
            _cache.clear();
            [self _freeOnLoadQueue:old];
            [self _setState:AFMNeuralSelectorStateFailed];
        }
        os_log_error(AFMNeuralLog(), "R-walk unavailable, using the stock walk: %{public}@", result.error ?: @"unknown error");
        return;
    }
    AFMLoadedScorer old = _loaded;
    _loaded = result.loaded;
    _scorer = result.loaded.scorer;
    _cache.clear();
    _loadedKey = key;
    _modelIdentifier = identifier;
    if (request == nil) {
        _personalStatus = AFMPersonalModelStatusFactory;
        _loadedPersonalVersion = 0;
        _personalError = nil;
    } else if (result.personalApplied) {
        _personalStatus = AFMPersonalModelStatusActive;
        _loadedPersonalVersion = request.version;
        _personalError = nil;
    } else {
        _personalStatus = result.personalMismatch ? AFMPersonalModelStatusMismatch : AFMPersonalModelStatusUnusable;
        _loadedPersonalVersion = 0;
        _personalError = [result.personalError copy];
        // Logged once: the same key is not loaded again (design D11).
        os_log_error(AFMNeuralLog(), "personalized version %ld not used, factory weights: %{public}@", (long)request.version, (result.personalError != nil ? result.personalError : @"unknown"));
    }
    // The old scorer is only used on main, synchronously; after the swap
    // nothing references it.
    [self _freeOnLoadQueue:old];
    [AFMPreciseController.sharedController setPersonalVersion:(result.personalApplied ? request : nil)];
    if (_state == AFMNeuralSelectorStateLoading) {
        [self _setState:AFMNeuralSelectorStateReady];
        os_log_info(AFMNeuralLog(), "R-walk model ready in %.0f ms", ms);
    } else {
        [NSNotificationCenter.defaultCenter postNotificationName:AFMNeuralSelectorStateDidChangeNotification object:self];
    }
    // Something else may have been selected while this one loaded.
    [self _reloadIfNeeded];
}

- (void)waitForLoadQueueForTesting
{
    dispatch_sync(_loadQueue, ^ {});
}

- (void)_setState:(AFMNeuralSelectorState)state
{
    if (_state == state) {
        return;
    }
    _state = state;
    if (NSThread.isMainThread) {
        [NSNotificationCenter.defaultCenter postNotificationName:AFMNeuralSelectorStateDidChangeNotification object:self];
    } else {
        dispatch_async(dispatch_get_main_queue(), ^{
            [NSNotificationCenter.defaultCenter postNotificationName:AFMNeuralSelectorStateDidChangeNotification object:self];
        });
    }
}

- (Formosa::Gramambular2::ReadingGrid::WalkResult)walkGrid:(Formosa::Gramambular2::ReadingGrid &)grid
{
    return [self walkGrid:grid deferBreakerRecording:NO usedNeural:nullptr];
}

- (Formosa::Gramambular2::ReadingGrid::WalkResult)walkGrid:(Formosa::Gramambular2::ReadingGrid &)grid
                                      deferBreakerRecording:(BOOL)deferBreakerRecording
                                                 usedNeural:(BOOL *)outUsedNeural
{
    // A measurement left pending by the previous key event (its hook did not
    // run) is recorded before anything else, so it counts exactly once.
    [self flushPendingBreakerRecordingAddingMilliseconds:0];
    if (outUsedNeural != nullptr) {
        *outUsedNeural = NO;
    }
    if (!_enabled || _alpha == 0 || _state != AFMNeuralSelectorStateReady || _scorer == nullptr || grid.length() == 0 || grid.length() > kAFMNeuralMaxSyllables) {
        return grid.walk();
    }

    // The table is keyed on (readings, characters per position); compute the
    // latter once and use it for the lookup, the forward pass and the store.
    auto t0 = std::chrono::steady_clock::now();
    McBopomofo::CharsAtPositions chars = grid.charsAtPositions();
    const auto *scores = _cache.lookup(grid.readings(), chars);
    bool miss = scores == nullptr;
    if (miss) {
        McBopomofo::NeuralWalkOptions options;
        options.alpha = _alpha;
        McBopomofo::NeuralScoreCache::Scores fresh;
        try {
            fresh = McBopomofo::ComputeCharScores(grid, *_scorer, options, chars);
        } catch (...) {
            fresh = McBopomofo::NeuralScoreCache::Scores();
        }
        ++_forwardCount;
        // Empty tables (failed forward) are not stored, so the next walk retries.
        scores = _cache.store(grid.readings(), std::move(chars), std::move(fresh));
    }

    bool usedNeural = false;
    auto result = scores != nullptr ? McBopomofo::WalkWithScores(grid, *scores, _alpha, &usedNeural) : grid.walk();
    if (usedNeural) {
        ++_neuralWalkCount;
    }
    if (outUsedNeural != nullptr) {
        *outUsedNeural = usedNeural;
    }

    if (miss) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (deferBreakerRecording) {
            _hasPendingBreakerMs = YES;
            _pendingBreakerMs = ms;
        } else {
            [self _recordBreakerMilliseconds:ms];
        }
    }
    return result;
}

- (void)flushPendingBreakerRecordingAddingMilliseconds:(double)extraMilliseconds
{
    if (!_hasPendingBreakerMs) {
        return;
    }
    _hasPendingBreakerMs = NO;
    [self _recordBreakerMilliseconds:_pendingBreakerMs + extraMilliseconds];
}

- (void)_recordBreakerMilliseconds:(double)ms
{
    ++_breakerRecordCount;
    if (_circuitBreakerDisabledForTesting) {
        return;
    }
    _breaker.record(ms);
    if (_breaker.tripped() && _state != AFMNeuralSelectorStateTripped) {
        [self _setState:AFMNeuralSelectorStateTripped];
        _cache.clear();
        os_log_error(AFMNeuralLog(), "R-walk disabled for this process: %d consecutive keystrokes over %.0f ms", McBopomofo::NeuralCircuitBreaker::kConsecutive, McBopomofo::NeuralCircuitBreaker::kThresholdMs);
    }
}

- (std::optional<AFMPrecisePayload>)precisePayloadForGrid:(Formosa::Gramambular2::ReadingGrid &)grid
                                            displayedText:(const std::string &)displayedText
                                                   config:(const McBopomofo::PreciseConfig &)config
{
    if (!_enabled || _alpha == 0 || _state != AFMNeuralSelectorStateReady || _scorer == nullptr || grid.length() == 0 || grid.length() > kAFMNeuralMaxSyllables) {
        return std::nullopt;
    }
    // Same table as the walk: no forward pass here.
    const auto *scores = _cache.lookup(grid);
    if (scores == nullptr) {
        return std::nullopt;
    }
    ++_precisePayloadCount;
    AFMPrecisePayload payload;
    payload.candidates = McBopomofo::PreciseCandidates(grid, *scores, config);
    McBopomofo::FilterFixedPositions(&payload.candidates, McBopomofo::FixedPositionsOf(grid));
    size_t rank = McBopomofo::MoveDisplayedToFront(&payload.candidates, displayedText);
    if (rank == std::string::npos) {
        return std::nullopt;
    }
    // MoveDisplayedToFront also moves a lower-ranked text; only accept that for
    // a tie with R's best (then rank 1 is the former rank 0).
    if (rank > 0 && std::fabs(payload.candidates.rScores[0] - payload.candidates.rScores[1]) > 1e-9) {
        return std::nullopt;
    }
    auto gate = McBopomofo::PreciseGate(grid.readings(), payload.candidates, config);
    if (!gate.has_value()) {
        return std::nullopt;
    }
    payload.gate = std::move(*gate);
    return payload;
}

@end
