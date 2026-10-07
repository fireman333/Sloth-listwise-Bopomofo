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
#import "McBopomofo-Swift.h"

#import <malloc/malloc.h>
#import <os/log.h>

#import <chrono>
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

// Production loader. Runs on the load queue. Returns a heap-allocated scorer
// wrapping a heap-allocated SlothRescorer, both intentionally never freed, or
// nullptr with *error set.
static McBopomofo::NeuralScorer *AFMLoadSlothScorer(NSString *modelDirectory, NSString **error)
{
    if (modelDirectory == nil) {
        *error = @"no model directory configured";
        return nullptr;
    }
    if (&OrtGetApiBase == nullptr) {
        *error = @"onnxruntime is not available (libonnxruntime.1.dylib missing)";
        return nullptr;
    }
    BOOL isDirectory = NO;
    if (![[NSFileManager defaultManager] fileExistsAtPath:modelDirectory isDirectory:&isDirectory] || !isDirectory) {
        *error = [NSString stringWithFormat:@"model directory missing: %@", modelDirectory];
        return nullptr;
    }
    const OrtApiBase *apiBase = OrtGetApiBase();
    const OrtApi *api = apiBase != nullptr ? apiBase->GetApi(ORT_API_VERSION) : nullptr;
    if (api == nullptr) {
        *error = [NSString stringWithFormat:@"onnxruntime does not provide API version %d", ORT_API_VERSION];
        return nullptr;
    }
    Ort::InitApi(api);

    auto *model = new McBopomofo::SlothRescorer(std::string(modelDirectory.fileSystemRepresentation));
    if (!model->ready()) {
        delete model;
        *error = [NSString stringWithFormat:@"failed to load the R model from %@", modelDirectory];
        return nullptr;
    }
    return new McBopomofo::NeuralScorer(McBopomofo::MakeNeuralScorer(model));
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

typedef McBopomofo::NeuralScorer * (^AFMNeuralLoader)(NSString **error);

static AFMNeuralSelector *gSharedSelector = nil;

@interface AFMNeuralSelector ()
- (instancetype)_initWithModelDirectory:(nullable NSString *)modelDirectory
                                 loader:(AFMNeuralLoader)loader NS_DESIGNATED_INITIALIZER;
@end

@implementation AFMNeuralSelector {
    AFMNeuralLoader _loader;
    dispatch_queue_t _loadQueue;
    // Owned and never freed, so no onnxruntime teardown runs at process exit.
    McBopomofo::NeuralScorer *_scorer;
    McBopomofo::NeuralScoreCache _cache;
    McBopomofo::NeuralCircuitBreaker _breaker;
}

@synthesize state = _state;
@synthesize enabled = _enabled;
@synthesize alpha = _alpha;
@synthesize modelDirectory = _modelDirectory;
@synthesize lastLoadMilliseconds = _lastLoadMilliseconds;
@synthesize forwardCount = _forwardCount;
@synthesize neuralWalkCount = _neuralWalkCount;

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
    return [self _initWithModelDirectory:directory loader:^McBopomofo::NeuralScorer *(NSString **error) {
        return AFMLoadSlothScorer(directory, error);
    }];
}

- (instancetype)initWithScoreBlockForTesting:(AFMNeuralScoreBlock)scoreBlock loadDelay:(NSTimeInterval)loadDelay
{
    AFMNeuralScoreBlock block = [scoreBlock copy];
    return [self _initWithModelDirectory:nil loader:^McBopomofo::NeuralScorer *(NSString **error) {
        if (loadDelay > 0) {
            [NSThread sleepForTimeInterval:loadDelay];
        }
        return AFMMakeBlockScorer(block);
    }];
}

- (void)syncWithPreferences
{
    _enabled = Preferences.neuralWalkEnabled;
    _alpha = Preferences.neuralWalkAlpha;
    [self startLoadingIfNeeded];
}

- (void)startLoadingIfNeeded
{
    if (!_enabled || _state != AFMNeuralSelectorStateDisabled) {
        return;
    }
    _state = AFMNeuralSelectorStateLoading;
    AFMNeuralLoader loader = _loader;
    dispatch_async(_loadQueue, ^{
        auto t0 = std::chrono::steady_clock::now();
        NSString *error = nil;
        McBopomofo::NeuralScorer *scorer = nullptr;
        @try {
            scorer = loader(&error);
        } @catch (NSException *exception) {
            scorer = nullptr;
            error = exception.reason;
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (scorer != nullptr) {
            // Hand back the transient memory onnxruntime used while loading.
            malloc_zone_pressure_relief(NULL, 0);
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            [self _publishScorer:scorer error:error milliseconds:ms];
        });
    });
}

- (void)_publishScorer:(McBopomofo::NeuralScorer *)scorer error:(NSString *)error milliseconds:(double)ms
{
    _lastLoadMilliseconds = ms;
    if (_state != AFMNeuralSelectorStateLoading) {
        // Never publish over a state that changed meanwhile. The scorer, if
        // any, is leaked on purpose.
        return;
    }
    if (scorer == nullptr) {
        _state = AFMNeuralSelectorStateFailed;
        os_log_error(AFMNeuralLog(), "R-walk unavailable, using the stock walk: %{public}@", error ?: @"unknown error");
        return;
    }
    _scorer = scorer;
    _cache.clear();
    _state = AFMNeuralSelectorStateReady;
    os_log_info(AFMNeuralLog(), "R-walk model ready in %.0f ms", ms);
}

- (Formosa::Gramambular2::ReadingGrid::WalkResult)walkGrid:(Formosa::Gramambular2::ReadingGrid &)grid
{
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

    if (miss) {
        _breaker.record(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        if (_breaker.tripped()) {
            _state = AFMNeuralSelectorStateTripped;
            _cache.clear();
            os_log_error(AFMNeuralLog(), "R-walk disabled for this process: %d consecutive keystrokes over %.0f ms", McBopomofo::NeuralCircuitBreaker::kConsecutive, McBopomofo::NeuralCircuitBreaker::kThresholdMs);
        }
    }
    return result;
}

@end
