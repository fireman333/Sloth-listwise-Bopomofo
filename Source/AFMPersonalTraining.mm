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

#import "AFMPersonalTraining.h"
#import "AFMLearningStore.h"
#import "AFMNeuralSelector.h"
#import "McBopomofo-Swift.h"

#import <IOKit/IOKitLib.h>
#import <IOKit/ps/IOPSKeys.h>
#import <IOKit/ps/IOPowerSources.h>
#import <os/log.h>

#include <atomic>
#include <cerrno>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "Personalize/onnx_initializers.h"
#include "Personalize/personal_weights.h"
#include "Personalize/pool_reader.h"

extern char **environ;

NSNotificationName const AFMPersonalTrainingControllerDidChangeNotification = @"AFMPersonalTrainingControllerDidChangeNotification";
NSString *const AFMPersonalDirectoryName = @"r-personal";

namespace pz = McBopomofo::Personalize;

static NSString *AFMPersonalBundleIdentifier(void)
{
    NSString *identifier = NSBundle.mainBundle.bundleIdentifier;
    return identifier != nil ? identifier : @"io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo";
}

static os_log_t AFMPersonalLog(void)
{
    static os_log_t log;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        log = os_log_create(AFMPersonalBundleIdentifier().UTF8String, "Personalize");
    });
    return log;
}

#pragma mark - File helpers (store queue)

static BOOL AFMMakeDirectory700(NSString *path)
{
    NSError *error = nil;
    if (![NSFileManager.defaultManager createDirectoryAtPath:path withIntermediateDirectories:YES attributes:@{ NSFilePosixPermissions : @(0700) } error:&error]) {
        return NO;
    }
    chmod(path.fileSystemRepresentation, 0700);
    return YES;
}

static NSDictionary *_Nullable AFMReadJSONFile(NSString *path)
{
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (data == nil) {
        return nil;
    }
    id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    return [object isKindOfClass:NSDictionary.class] ? object : nil;
}

// .tmp (0600) + rename.
static BOOL AFMWriteJSONFileAtomically(NSString *path, NSDictionary *object)
{
    NSData *data = [NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingSortedKeys error:nil];
    if (data == nil || !AFMMakeDirectory700(path.stringByDeletingLastPathComponent)) {
        return NO;
    }
    NSString *tmp = [path stringByAppendingString:@".tmp"];
    int fd = open(tmp.fileSystemRepresentation, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
        return NO;
    }
    fchmod(fd, 0600);
    const char *p = static_cast<const char *>(data.bytes);
    size_t left = data.length;
    BOOL ok = YES;
    while (left > 0) {
        ssize_t n = write(fd, p, left);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            ok = NO;
            break;
        }
        p += n;
        left -= static_cast<size_t>(n);
    }
    close(fd);
    if (!ok || rename(tmp.fileSystemRepresentation, path.fileSystemRepresentation) != 0) {
        unlink(tmp.fileSystemRepresentation);
        return NO;
    }
    return YES;
}

static void AFMRemoveItem(NSString *path)
{
    [NSFileManager.defaultManager removeItemAtPath:path error:nil];
}

static NSString *AFMStringValue(id value)
{
    return [value isKindOfClass:NSString.class] ? value : nil;
}

static NSNumber *AFMNumberValue(id value)
{
    return [value isKindOfClass:NSNumber.class] ? value : nil;
}

#pragma mark - Evaluation configuration

NSString *AFMFactoryModelSHA256(NSString *modelDirectory)
{
    static NSMutableDictionary<NSString *, NSArray *> *cache;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        cache = [NSMutableDictionary dictionary];
    });
    NSString *path = [modelDirectory stringByAppendingPathComponent:@"onnx/model_quantized.onnx"];
    NSDictionary *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:path error:nil];
    if (attributes == nil) {
        return nil;
    }
    NSArray *key = @[ @(attributes.fileSize), @(attributes.fileModificationDate.timeIntervalSince1970) ];
    @synchronized(cache) {
        NSArray *entry = cache[path];
        if (entry != nil && [entry[0] isEqual:key]) {
            return entry[1];
        }
    }
    pz::Sha256Digest digest;
    std::string error;
    if (!pz::Sha256File(path.fileSystemRepresentation, &digest, &error)) {
        return nil;
    }
    NSString *hex = @(pz::ToHex(digest).c_str());
    @synchronized(cache) {
        cache[path] = @[ key, hex ];
    }
    return hex;
}

static NSString *AFMPackDirectory(NSString *modelDirectory)
{
    return [modelDirectory stringByAppendingPathComponent:@"personalize"];
}

NSString *AFMPersonalPackVersion(NSString *modelDirectory)
{
    NSDictionary *pack = AFMReadJSONFile([AFMPackDirectory(modelDirectory) stringByAppendingPathComponent:@"pack.json"]);
    if (pack == nil) {
        return nil;
    }
    NSString *version = AFMStringValue(pack[@"pack_version"]);
    return version != nil ? version : @"";
}

NSDictionary *AFMCurrentEvalConfig(NSString *factorySHA, NSString *packVersion, double alpha, BOOL acceptYan)
{
    // Same keys as the helper's report.json eval_config (design D10 note).
    return @{
        @"factory_sha256" : (factorySHA != nil ? factorySHA : @""),
        @"alpha" : @(alpha),
        @"accept_yan_for_ai" : @(acceptYan),
        @"pack_version" : (packVersion != nil ? packVersion : @""),
        @"calibration_version" : (packVersion != nil ? packVersion : @""),
    };
}

BOOL AFMEvalConfigMatches(NSDictionary *a, NSDictionary *b)
{
    if (a == nil || b == nil) {
        return NO;
    }
    for (NSString *key in @[ @"factory_sha256", @"pack_version", @"calibration_version" ]) {
        NSString *x = AFMStringValue(a[key]);
        NSString *y = AFMStringValue(b[key]);
        if (x == nil || y == nil || x.length == 0 || ![x isEqualToString:y]) {
            return NO;
        }
    }
    NSNumber *alphaA = AFMNumberValue(a[@"alpha"]);
    NSNumber *alphaB = AFMNumberValue(b[@"alpha"]);
    if (alphaA == nil || alphaB == nil || std::fabs(alphaA.doubleValue - alphaB.doubleValue) > 1e-9) {
        return NO;
    }
    NSNumber *yanA = AFMNumberValue(a[@"accept_yan_for_ai"]);
    NSNumber *yanB = AFMNumberValue(b[@"accept_yan_for_ai"]);
    return yanA != nil && yanB != nil && yanA.boolValue == yanB.boolValue;
}

NSString *AFMPreciseOverlayRun(NSString *path)
{
    NSString *text = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    for (NSString *raw in [text componentsSeparatedByString:@"\n"]) {
        NSString *line = [raw stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
        if (![line hasPrefix:@"run"]) {
            continue;
        }
        NSRange eq = [line rangeOfString:@"="];
        if (eq.location == NSNotFound) {
            continue;
        }
        NSString *key = [[line substringToIndex:eq.location] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
        if ([key isEqualToString:@"run"]) {
            return [[line substringFromIndex:eq.location + 1] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
        }
    }
    return nil;
}

#pragma mark - Testing helpers

static bool AFMDigestFromHex(NSString *hex, pz::Sha256Digest *out)
{
    if (hex.length != 64) {
        return false;
    }
    for (size_t i = 0; i < 32; ++i) {
        unsigned int byte = 0;
        if (sscanf([hex substringWithRange:NSMakeRange(i * 2, 2)].UTF8String, "%2x", &byte) != 1) {
            return false;
        }
        (*out)[i] = static_cast<uint8_t>(byte);
    }
    return true;
}

BOOL AFMWriteFakeWeightsFileForTesting(NSString *path, NSString *factorySHA, NSInteger count)
{
    pz::PersonalWeights weights;
    if (!AFMDigestFromHex(factorySHA, &weights.factorySha)) {
        return NO;
    }
    for (NSInteger i = 0; i < count; ++i) {
        pz::QuantizedLinear linear;
        linear.name = "val_" + std::to_string(i);
        linear.in = 2;
        linear.out = 2;
        linear.scales = { 0.5f, 0.25f };
        linear.values = { 1, -2, 3, -4 };
        weights.tensors.push_back(std::move(linear));
    }
    std::string error;
    return pz::WriteWeightsFile(path.fileSystemRepresentation, weights, &error);
}

BOOL AFMWriteFactoryWeightsFileForTesting(NSString *modelDirectory, NSString *path, double scaleFactor)
{
    NSString *onnx = [modelDirectory stringByAppendingPathComponent:@"onnx/model_quantized.onnx"];
    std::string bytes, error;
    if (!pz::ReadFileBytes(onnx.fileSystemRepresentation, &bytes, &error)) {
        return NO;
    }
    pz::OnnxModelInfo info;
    pz::PersonalWeights weights;
    if (!pz::ScanOnnxModel(bytes, &info, &error) || !pz::FactoryLinears(info, &weights.tensors, &error)) {
        return NO;
    }
    weights.factorySha = pz::Sha256(bytes.data(), bytes.size());
    for (auto &linear : weights.tensors) {
        for (auto &scale : linear.scales) {
            scale = static_cast<float>(scale * scaleFactor);
        }
    }
    return pz::WriteWeightsFile(path.fileSystemRepresentation, weights, &error);
}

#pragma mark - Probes

static double AFMSystemIdleSeconds(void)
{
    io_iterator_t iterator = 0;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOHIDSystem"), &iterator) != KERN_SUCCESS) {
        return 0;
    }
    double seconds = 0;
    io_registry_entry_t entry = IOIteratorNext(iterator);
    if (entry != 0) {
        CFTypeRef value = IORegistryEntryCreateCFProperty(entry, CFSTR("HIDIdleTime"), kCFAllocatorDefault, 0);
        if (value != NULL) {
            int64_t ns = 0;
            if (CFGetTypeID(value) == CFNumberGetTypeID() && CFNumberGetValue((CFNumberRef)value, kCFNumberSInt64Type, &ns)) {
                seconds = ns / 1e9;
            }
            CFRelease(value);
        }
        IOObjectRelease(entry);
    }
    IOObjectRelease(iterator);
    return seconds;
}

@implementation AFMPersonalSystemProbes
@synthesize onACPower = _onACPower;
@synthesize lowPowerMode = _lowPowerMode;
@synthesize thermalPressure = _thermalPressure;
@synthesize idleSeconds = _idleSeconds;

+ (instancetype)systemProbes
{
    AFMPersonalSystemProbes *probes = [[AFMPersonalSystemProbes alloc] init];
    probes.onACPower = ^BOOL {
        CFTypeRef info = IOPSCopyPowerSourcesInfo();
        if (info == NULL) {
            return YES;
        }
        CFStringRef type = IOPSGetProvidingPowerSourceType(info);
        BOOL ac = type == NULL || CFStringCompare(type, CFSTR(kIOPMACPowerKey), 0) == kCFCompareEqualTo;
        CFRelease(info);
        return ac;
    };
    probes.lowPowerMode = ^BOOL {
        return NSProcessInfo.processInfo.lowPowerModeEnabled;
    };
    probes.thermalPressure = ^BOOL {
        return NSProcessInfo.processInfo.thermalState > NSProcessInfoThermalStateFair;
    };
    probes.idleSeconds = ^double {
        return AFMSystemIdleSeconds();
    };
    return probes;
}

@end

#pragma mark - Version request

@implementation AFMPersonalVersionRequest
@synthesize version = _version;
@synthesize directory = _directory;
@synthesize runIdentifier = _runIdentifier;
@synthesize evalConfig = _evalConfig;
@synthesize generation = _generation;
@synthesize created = _created;

- (instancetype)initWithVersion:(NSInteger)version directory:(NSString *)directory runIdentifier:(NSString *)runIdentifier evalConfig:(NSDictionary *)evalConfig generation:(uint64_t)generation created:(double)created
{
    self = [super init];
    if (self) {
        _version = version;
        _directory = [directory copy];
        _runIdentifier = [runIdentifier copy];
        _evalConfig = [evalConfig copy];
        _generation = generation;
        _created = created;
    }
    return self;
}

- (BOOL)isEqual:(id)object
{
    if (![object isKindOfClass:AFMPersonalVersionRequest.class]) {
        return NO;
    }
    AFMPersonalVersionRequest *other = object;
    return other.version == _version && other.generation == _generation && [other.directory isEqualToString:_directory] && [other.runIdentifier isEqualToString:_runIdentifier];
}

- (NSUInteger)hash
{
    return (NSUInteger)_version ^ (NSUInteger)_generation ^ _directory.hash;
}

@end

#pragma mark - Controller

/// One helper process (main thread).
@interface AFMPersonalRun : NSObject
@property (copy, nonatomic) NSString *runIdentifier;
@property (nonatomic) pid_t pid;
@property (nonatomic) uint64_t generation;
@property (nonatomic) BOOL automatic;
@property (copy, nonatomic) NSString *stagingDirectory;
@property (copy, nonatomic, nullable) NSString *killReason;
@property (nonatomic) double started;
@property (nonatomic) NSInteger eligibleAtStart;
@property (copy, nonatomic) NSDictionary *evalConfig;
@end

@implementation AFMPersonalRun
@synthesize runIdentifier = _runIdentifier;
@synthesize pid = _pid;
@synthesize generation = _generation;
@synthesize automatic = _automatic;
@synthesize stagingDirectory = _stagingDirectory;
@synthesize killReason = _killReason;
@synthesize started = _started;
@synthesize eligibleAtStart = _eligibleAtStart;
@synthesize evalConfig = _evalConfig;
@end

static AFMPersonalTrainingController *gSharedController = nil;

@implementation AFMPersonalTrainingController {
    NSMutableDictionary *_manifest;
    std::atomic<uint64_t> _queueGeneration;
    AFMPersonalRun *_run;
    NSInteger _eligibleCount;
    unsigned long long _scannedPoolSize;
    BOOL _scanPending;
    NSMutableArray<dispatch_block_t> *_scanWaiters;
    NSMutableArray<dispatch_block_t> *_clearWaiters;
    BOOL _clearing;
    NSUInteger _pendingMainWork;
    // Store queue only.
    BOOL _vocabLoaded;
    BOOL _vocabOK;
    pz::Vocab _vocab;
    NSBackgroundActivityScheduler *_scheduler;
    NSUInteger _spawnCount;
    // Guarded by @synchronized(self): read on the store queue.
    NSMutableSet<NSString *> *_protectedRunIdentifiers;
}

@synthesize memoryDirectory = _memoryDirectory;
@synthesize modelDirectory = _modelDirectory;
@synthesize eligibleCount = _eligibleCount;
@synthesize packAvailable = _packAvailable;
@synthesize factorySHA = _factorySHA;
@synthesize packVersion = _packVersion;
@synthesize scheduler = _scheduler;
@synthesize helperPath = _helperPath;
@synthesize clock = _clock;
@synthesize probes = _probes;
@synthesize learningEnabledCheck = _learningEnabledCheck;
@synthesize runTimeLimit = _runTimeLimit;
@synthesize killGracePeriod = _killGracePeriod;
@synthesize minimumEligible = _minimumEligible;
@synthesize minimumNewEligible = _minimumNewEligible;
@synthesize automaticInterval = _automaticInterval;
@synthesize minimumIdle = _minimumIdle;
@synthesize helperThreads = _helperThreads;
@synthesize failurePauseCount = _failurePauseCount;
@synthesize generation = _generation;
@synthesize runDidFinishHandler = _runDidFinishHandler;

+ (AFMPersonalTrainingController *)sharedController
{
    if (gSharedController == nil) {
        NSString *helper = [NSBundle.mainBundle pathForAuxiliaryExecutable:@"AFMPersonalTrainer"];
        gSharedController = [[AFMPersonalTrainingController alloc] initWithMemoryDirectory:UserPhraseLocationHelper.learningDataDirectory
                                                                           modelDirectory:UserPhraseLocationHelper.rWalkModelDirectory
                                                                               helperPath:helper];
    }
    return gSharedController;
}

+ (void)setSharedController:(AFMPersonalTrainingController *)controller
{
    gSharedController = controller;
}

+ (void)resetForTesting
{
    [gSharedController stopScheduler];
    gSharedController = [[AFMPersonalTrainingController alloc] initWithMemoryDirectory:UserPhraseLocationHelper.learningDataDirectory
                                                                       modelDirectory:UserPhraseLocationHelper.rWalkModelDirectory
                                                                           helperPath:nil];
}

- (instancetype)initWithMemoryDirectory:(NSString *)memoryDirectory modelDirectory:(NSString *)modelDirectory helperPath:(NSString *)helperPath
{
    self = [super init];
    if (self) {
        _memoryDirectory = [memoryDirectory copy];
        _modelDirectory = [modelDirectory copy];
        _helperPath = [helperPath copy];
        _probes = [AFMPersonalSystemProbes systemProbes];
        _runTimeLimit = 600;
        _killGracePeriod = 5;
        _minimumEligible = 1000;
        _minimumNewEligible = 200;
        _automaticInterval = 24 * 3600;
        _minimumIdle = 600;
        _helperThreads = 4;
        _failurePauseCount = 3;
        _eligibleCount = -1;
        _queueGeneration = 0;
        _scanWaiters = [NSMutableArray array];
        _clearWaiters = [NSMutableArray array];
        // The manifest is tiny; read once here (launch / tests, not the key path).
        NSDictionary *manifest = AFMReadJSONFile([self _manifestPath]);
        _manifest = manifest != nil ? [manifest mutableCopy] : [NSMutableDictionary dictionary];
        _generation = 0;
        _packAvailable = [NSFileManager.defaultManager fileExistsAtPath:[AFMPackDirectory(_modelDirectory) stringByAppendingPathComponent:@"pack.json"]];
    }
    return self;
}

#pragma mark Paths and basics

- (NSString *)personalDirectory
{
    return [_memoryDirectory stringByAppendingPathComponent:AFMPersonalDirectoryName];
}

- (NSString *)_manifestPath
{
    return [self.personalDirectory stringByAppendingPathComponent:@"manifest.json"];
}

- (NSString *)_versionDirectory:(NSInteger)version
{
    return [self.personalDirectory stringByAppendingPathComponent:[NSString stringWithFormat:@"v%ld", (long)version]];
}

- (NSString *)_stagingDirectory:(NSString *)run
{
    return [[self.personalDirectory stringByAppendingPathComponent:@"staging"] stringByAppendingPathComponent:run];
}

- (double)_now
{
    return _clock ? _clock() : NSDate.date.timeIntervalSince1970;
}

- (BOOL)_learningEnabled
{
    return _learningEnabledCheck ? _learningEnabledCheck() : Preferences.learningEnabled;
}

- (void)_notify
{
    [NSNotificationCenter.defaultCenter postNotificationName:AFMPersonalTrainingControllerDidChangeNotification object:self];
}

- (dispatch_queue_t)_ioQueue
{
    return AFMLearningStore.sharedStore.storeQueue;
}

/// Runs `work` on the store queue and `completion` on main, both only if the
/// personalization generation did not change since now.
- (void)_onQueue:(id _Nullable (^)(void))work then:(void (^_Nullable)(id _Nullable result))completion
{
    [self _onQueue:work then:completion discarded:nil];
}

/// Same; `discarded` runs on main instead of `completion` when a clear
/// bumped the generation meanwhile.
- (void)_onQueue:(id _Nullable (^)(void))work then:(void (^_Nullable)(id _Nullable result))completion discarded:(dispatch_block_t _Nullable)discarded
{
    uint64_t generation = _generation;
    ++_pendingMainWork;
    dispatch_async([self _ioQueue], ^{
        id result = nil;
        if (self->_queueGeneration.load() == generation) {
            result = work();
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            --self->_pendingMainWork;
            if (self->_generation == generation) {
                if (completion != nil) {
                    completion(result);
                }
            } else if (discarded != nil) {
                discarded();
            }
        });
    });
}

#pragma mark Manifest (main thread copy; written on the store queue)

- (void)_writeManifest
{
    _manifest[@"format"] = @1;
    _manifest[@"generation"] = @(_generation);
    NSDictionary *copy = [_manifest copy];
    NSString *path = [self _manifestPath];
    uint64_t generation = _generation;
    dispatch_async([self _ioQueue], ^{
        if (self->_queueGeneration.load() == generation) {
            AFMWriteJSONFileAtomically(path, copy);
        }
    });
}

- (NSNumber *)activeVersion
{
    return AFMNumberValue(_manifest[@"active"]);
}

- (NSNumber *)previousVersion
{
    return AFMNumberValue(_manifest[@"previous"]);
}

- (BOOL)paused
{
    return [AFMNumberValue(_manifest[@"paused"]) boolValue];
}

- (BOOL)failurePaused
{
    return [AFMNumberValue(_manifest[@"failure_paused"]) boolValue];
}

- (NSInteger)consecutiveFailures
{
    return [AFMNumberValue(_manifest[@"failures"]) integerValue];
}

- (NSDictionary *)lastRun
{
    id value = _manifest[@"last_run"];
    return [value isKindOfClass:NSDictionary.class] ? value : nil;
}

- (NSDictionary *)_versionEntry:(NSInteger)version
{
    NSDictionary *versions = _manifest[@"versions"];
    if (![versions isKindOfClass:NSDictionary.class]) {
        return nil;
    }
    id entry = versions[[NSString stringWithFormat:@"%ld", (long)version]];
    return [entry isKindOfClass:NSDictionary.class] ? entry : nil;
}

- (double)createdTimeOfVersion:(NSInteger)version
{
    return [AFMNumberValue([self _versionEntry:version][@"created"]) doubleValue];
}

- (NSDictionary *)currentEvalConfig
{
    return AFMCurrentEvalConfig(_factorySHA, _packVersion, Preferences.neuralWalkAlpha, Preferences.acceptYanReadingForAi);
}

- (AFMPersonalVersionRequest *)activeVersionRequest
{
    NSNumber *active = self.activeVersion;
    if (active == nil || _clearing) {
        return nil;
    }
    NSDictionary *entry = [self _versionEntry:active.integerValue];
    if (entry == nil) {
        return nil;
    }
    NSDictionary *config = [entry[@"eval_config"] isKindOfClass:NSDictionary.class] ? entry[@"eval_config"] : @{};
    return [[AFMPersonalVersionRequest alloc] initWithVersion:active.integerValue
                                                    directory:[self _versionDirectory:active.integerValue]
                                                runIdentifier:(AFMStringValue(entry[@"run"]) != nil ? AFMStringValue(entry[@"run"]) : @"")
                                                   evalConfig:config
                                                   generation:_generation
                                                      created:[AFMNumberValue(entry[@"created"]) doubleValue]];
}

#pragma mark Refresh

- (void)refreshWithCompletion:(dispatch_block_t)completion
{
    NSString *modelDirectory = _modelDirectory;
    NSString *personal = self.personalDirectory;
    [self _onQueue:^id {
        NSMutableDictionary *result = [NSMutableDictionary dictionary];
        NSString *sha = AFMFactoryModelSHA256(modelDirectory);
        if (sha != nil) {
            result[@"sha"] = sha;
        }
        NSString *pack = AFMPersonalPackVersion(modelDirectory);
        if (pack != nil) {
            result[@"pack"] = pack;
        }
        // Staging dirs of crashed / killed runs (design D13); the run in
        // progress (read now, not when this block was queued) is kept.
        NSSet<NSString *> *running = [self _protectedRuns];
        NSString *staging = [personal stringByAppendingPathComponent:@"staging"];
        for (NSString *name in [NSFileManager.defaultManager contentsOfDirectoryAtPath:staging error:nil]) {
            if (![running containsObject:name]) {
                AFMRemoveItem([staging stringByAppendingPathComponent:name]);
            }
        }
        return result;
    } then:^(NSDictionary *result) {
        self->_factorySHA = result[@"sha"];
        self->_packVersion = result[@"pack"];
        self->_packAvailable = result[@"pack"] != nil;
        [self _handleFactoryChange];
        [self _deleteUnreferencedVersions];
        [self refreshEligibleCountWithCompletion:completion];
    }];
}

/// Design D13 "Factory update".
- (void)_handleFactoryChange
{
    if (_factorySHA == nil) {
        return;
    }
    NSString *known = AFMStringValue(_manifest[@"factory_sha"]);
    if (known != nil && ![known isEqualToString:_factorySHA]) {
        os_log(AFMPersonalLog(), "factory model changed: personalized versions deleted");
        for (NSString *key in @[ @"active", @"previous", @"versions", @"last_attempt", @"eligible_at_last_attempt", @"last_auto_attempt", @"paused", @"failure_paused", @"failures" ]) {
            [_manifest removeObjectForKey:key];
        }
        _manifest[@"factory_sha"] = _factorySHA;
        [self _writeManifest];
        [self _deleteUnreferencedVersions];
        [self _notify];
    } else if (known == nil) {
        _manifest[@"factory_sha"] = _factorySHA;
        [self _writeManifest];
    }
}

/// Deletes v<N> folders that are neither active nor previous.
- (void)_deleteUnreferencedVersions
{
    // Kept: active, previous and, after 「改回出廠模型」, the version that was
    // previous before it (spec: returning to factory keeps both versions).
    NSMutableSet<NSString *> *keep = [NSMutableSet set];
    for (NSNumber *n in @[ (self.activeVersion != nil ? self.activeVersion : @(-1)), (self.previousVersion != nil ? self.previousVersion : @(-1)), (AFMNumberValue(_manifest[@"reverted_previous"]) != nil ? AFMNumberValue(_manifest[@"reverted_previous"]) : @(-1)) ]) {
        if (n.integerValue >= 0) {
            [keep addObject:[NSString stringWithFormat:@"v%ld", (long)n.integerValue]];
        }
    }
    NSMutableDictionary *versions = [_manifest[@"versions"] isKindOfClass:NSDictionary.class] ? [_manifest[@"versions"] mutableCopy] : nil;
    BOOL changed = NO;
    for (NSString *key in versions.allKeys) {
        if (![keep containsObject:[@"v" stringByAppendingString:key]]) {
            [versions removeObjectForKey:key];
            changed = YES;
        }
    }
    if (changed) {
        _manifest[@"versions"] = versions;
        [self _writeManifest];
    }
    NSString *personal = self.personalDirectory;
    [self _onQueue:^id {
        for (NSString *name in [NSFileManager.defaultManager contentsOfDirectoryAtPath:personal error:nil]) {
            if (name.length > 1 && [name hasPrefix:@"v"] && [[name substringFromIndex:1] rangeOfCharacterFromSet:NSCharacterSet.decimalDigitCharacterSet.invertedSet].location == NSNotFound && ![keep containsObject:name]) {
                AFMRemoveItem([personal stringByAppendingPathComponent:name]);
            }
        }
        return nil;
    } then:nil];
}

// Store queue only.
- (BOOL)_loadVocabIfNeeded
{
    if (!_vocabLoaded) {
        _vocabLoaded = YES;
        std::string error;
        _vocabOK = pz::LoadModelVocab(_modelDirectory.fileSystemRepresentation, &_vocab, &error);
    }
    return _vocabOK;
}

- (void)refreshEligibleCountWithCompletion:(dispatch_block_t)completion
{
    if (completion != nil) {
        [_scanWaiters addObject:[completion copy]];
    }
    if (_scanPending) {
        return;
    }
    _scanPending = YES;
    NSString *pool = [_memoryDirectory stringByAppendingPathComponent:AFMLearningSentencePoolFileName];
    unsigned long long lastSize = _scannedPoolSize;
    NSInteger lastCount = _eligibleCount;
    uint64_t generation = _generation;
    ++_pendingMainWork;
    // On the store queue, after any queued append (design D7 / task 5.1).
    dispatch_async([self _ioQueue], ^{
        NSInteger count = 0;
        unsigned long long size = 0;
        NSDictionary *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:pool error:nil];
        size = attributes.fileSize;
        if (attributes == nil) {
            count = 0;
        } else if (size == lastSize && lastCount >= 0) {
            count = lastCount;
        } else if ([self _loadVocabIfNeeded]) {
            pz::PoolScan scan;
            std::string error;
            if (pz::ScanPool(pool.fileSystemRepresentation, self->_vocab, pz::SegmentOptions(), &scan, &error)) {
                count = (NSInteger)scan.segments.size();
            }
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            --self->_pendingMainWork;
            self->_scanPending = NO;
            if (self->_generation == generation) {
                BOOL changed = self->_eligibleCount != count;
                self->_eligibleCount = count;
                self->_scannedPoolSize = size;
                if (changed) {
                    [self _notify];
                }
            }
            NSArray<dispatch_block_t> *waiters = [self->_scanWaiters copy];
            [self->_scanWaiters removeAllObjects];
            for (dispatch_block_t waiter in waiters) {
                waiter();
            }
        });
    });
}

#pragma mark Starting

- (BOOL)isRunning
{
    return _run != nil;
}

// Any thread: runs whose staging dir must not be cleaned up (from the spawn
// until their result was handled).
- (NSSet<NSString *> *)_protectedRuns
{
    @synchronized(self) {
        return [_protectedRunIdentifiers copy];
    }
}

- (void)_protectRun:(NSString *)run protect:(BOOL)protect
{
    @synchronized(self) {
        if (_protectedRunIdentifiers == nil) {
            _protectedRunIdentifiers = [NSMutableSet set];
        }
        if (protect) {
            [_protectedRunIdentifiers addObject:run];
        } else {
            [_protectedRunIdentifiers removeObject:run];
        }
    }
}

- (NSString *)currentRunIdentifier
{
    return _run.runIdentifier;
}

- (BOOL)currentRunIsAutomatic
{
    return _run.automatic;
}

- (pid_t)helperProcessIdentifierForTesting
{
    return _run.pid;
}

- (NSUInteger)spawnCountForTesting
{
    return _spawnCount;
}

- (AFMPersonalStartResult)startManualRun
{
    if (![self _learningEnabled]) {
        return AFMPersonalStartResultLearningOff;
    }
    if (_clearing) {
        return AFMPersonalStartResultClearing;
    }
    if (_run != nil) {
        return AFMPersonalStartResultAlreadyRunning;
    }
    if (!_packAvailable) {
        return AFMPersonalStartResultPackMissing;
    }
    if (_eligibleCount < _minimumEligible) {
        [self refreshEligibleCountWithCompletion:nil];
        return AFMPersonalStartResultNotEnoughData;
    }
    AFMPersonalStartResult result = [self _spawnAutomatic:NO];
    if (result == AFMPersonalStartResultStarted) {
        // 「立即學習」 resumes automatic runs (design D13 / D15).
        [_manifest removeObjectForKey:@"paused"];
        [_manifest removeObjectForKey:@"failure_paused"];
        [self _writeManifest];
        [self _notify];
    }
    return result;
}

- (AFMPersonalBlockReason)automaticBlockReason
{
    if (![self _learningEnabled]) {
        return AFMPersonalBlockReasonLearningOff;
    }
    if (self.paused) {
        return AFMPersonalBlockReasonPaused;
    }
    if (self.failurePaused) {
        return AFMPersonalBlockReasonFailurePaused;
    }
    if (_run != nil || _clearing) {
        return AFMPersonalBlockReasonRunning;
    }
    if (!_packAvailable) {
        return AFMPersonalBlockReasonPackMissing;
    }
    if (_eligibleCount < _minimumEligible) {
        return AFMPersonalBlockReasonNotEnoughData;
    }
    NSNumber *lastAuto = AFMNumberValue(_manifest[@"last_auto_attempt"]);
    if (lastAuto != nil && [self _now] - lastAuto.doubleValue < _automaticInterval) {
        return AFMPersonalBlockReasonTooSoon;
    }
    // >= 200 new eligible segments since the last attempt, unless no attempt
    // was made yet on this factory model / configuration (design D6, D11).
    NSDictionary *lastAttempt = [_manifest[@"last_attempt"] isKindOfClass:NSDictionary.class] ? _manifest[@"last_attempt"] : nil;
    if (lastAttempt != nil && AFMEvalConfigMatches(lastAttempt[@"eval_config"], self.currentEvalConfig)) {
        NSInteger before = [AFMNumberValue(_manifest[@"eligible_at_last_attempt"]) integerValue];
        if (_eligibleCount - before < _minimumNewEligible) {
            return AFMPersonalBlockReasonNotEnoughNew;
        }
    }
    if (_probes.onACPower && !_probes.onACPower()) {
        return AFMPersonalBlockReasonOnBattery;
    }
    if (_probes.lowPowerMode && _probes.lowPowerMode()) {
        return AFMPersonalBlockReasonLowPowerMode;
    }
    if (_probes.thermalPressure && _probes.thermalPressure()) {
        return AFMPersonalBlockReasonThermal;
    }
    if (_probes.idleSeconds && _probes.idleSeconds() < _minimumIdle) {
        return AFMPersonalBlockReasonNotIdle;
    }
    return AFMPersonalBlockReasonNone;
}

- (BOOL)runAutomaticAttemptIfPossible
{
    if ([self automaticBlockReason] != AFMPersonalBlockReasonNone) {
        return NO;
    }
    return [self _spawnAutomatic:YES] == AFMPersonalStartResultStarted;
}

- (NSString *)_newRunIdentifier
{
    NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
    formatter.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
    formatter.dateFormat = @"yyyyMMdd-HHmmss";
    NSString *stamp = [formatter stringFromDate:[NSDate dateWithTimeIntervalSince1970:[self _now]]];
    return [NSString stringWithFormat:@"r%@-%04x", stamp, arc4random_uniform(0x10000)];
}

- (AFMPersonalStartResult)_spawnAutomatic:(BOOL)automatic
{
    NSString *helper = _helperPath;
    if (helper == nil || ![NSFileManager.defaultManager isExecutableFileAtPath:helper]) {
        os_log_error(AFMPersonalLog(), "personalization helper missing");
        return AFMPersonalStartResultHelperMissing;
    }
    if (!AFMMakeDirectory700(self.personalDirectory)) {
        return AFMPersonalStartResultSpawnFailed;
    }
    NSString *run = [self _newRunIdentifier];
    NSMutableArray<NSString *> *arguments = [NSMutableArray arrayWithArray:@[
        helper,
        @"--memory-dir", _memoryDirectory,
        @"--model-dir", _modelDirectory,
        @"--run-id", run,
        @"--generation", [NSString stringWithFormat:@"%llu", _generation],
        @"--mode", automatic ? @"auto" : @"manual",
        @"--accept-yan-for-ai", Preferences.acceptYanReadingForAi ? @"1" : @"0",
        @"--alpha", [NSString stringWithFormat:@"%.17g", Preferences.neuralWalkAlpha],
        @"--threads", [NSString stringWithFormat:@"%ld", (long)_helperThreads],
    ]];
    NSNumber *active = self.activeVersion;
    if (active != nil) {
        NSString *weights = [[self _versionDirectory:active.integerValue] stringByAppendingPathComponent:@"weights.bin"];
        if (AFMEvalConfigMatches([self _versionEntry:active.integerValue][@"eval_config"], self.currentEvalConfig)) {
            [arguments addObjectsFromArray:@[ @"--active-weights", weights ]];
        }
    }

    // posix_spawn: utility QoS (design D5 note, task 3.1), stdio to /dev/null,
    // every other descriptor closed, empty environment.
    std::vector<std::string> storage;
    for (NSString *a in arguments) {
        storage.emplace_back(a.fileSystemRepresentation);
    }
    std::vector<char *> argv;
    for (auto &s : storage) {
        argv.push_back(s.data());
    }
    argv.push_back(nullptr);
    char *envp[] = { nullptr };
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_CLOEXEC_DEFAULT);
    posix_spawnattr_set_qos_class_np(&attr, QOS_CLASS_UTILITY);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&actions, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", O_WRONLY, 0);
    pid_t pid = 0;
    int rc = posix_spawn(&pid, argv[0], &actions, &attr, argv.data(), envp);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    if (rc != 0) {
        os_log_error(AFMPersonalLog(), "personalization helper spawn failed: errno %d", rc);
        return AFMPersonalStartResultSpawnFailed;
    }
    setpriority(PRIO_PROCESS, pid, 5);
    ++_spawnCount;

    AFMPersonalRun *record = [[AFMPersonalRun alloc] init];
    record.runIdentifier = run;
    record.pid = pid;
    record.generation = _generation;
    record.automatic = automatic;
    record.stagingDirectory = [self _stagingDirectory:run];
    record.started = [self _now];
    record.eligibleAtStart = _eligibleCount;
    record.evalConfig = self.currentEvalConfig;
    _run = record;
    [self _protectRun:run protect:YES];

    _manifest[@"last_attempt"] = @{ @"run" : run, @"time" : @(record.started), @"mode" : automatic ? @"auto" : @"manual", @"eval_config" : record.evalConfig };
    _manifest[@"eligible_at_last_attempt"] = @(_eligibleCount);
    if (automatic) {
        _manifest[@"last_auto_attempt"] = @(record.started);
    }
    [self _writeManifest];
    os_log(AFMPersonalLog(), "personalization run started (%{public}s)", automatic ? "auto" : "manual");

    // Exit watcher: a dedicated thread blocks in waitpid (no race with an
    // early exit, unlike registering a kqueue source after the spawn).
    [NSThread detachNewThreadWithBlock:^{
        int status = 0;
        pid_t r;
        do {
            r = waitpid(pid, &status, 0);
        } while (r < 0 && errno == EINTR);
        dispatch_async(dispatch_get_main_queue(), ^{
            [self _helperDidExit:record status:status];
        });
    }];
    // Hard wall-clock limit (design D6).
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(_runTimeLimit * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        if (self->_run == record) {
            os_log_error(AFMPersonalLog(), "personalization run exceeded the time limit");
            [self _terminate:record reason:@"failed:timeout"];
        }
    });
    [self _notify];
    return AFMPersonalStartResultStarted;
}

- (void)_terminate:(AFMPersonalRun *)record reason:(NSString *)reason
{
    if (record.killReason == nil) {
        record.killReason = reason;
    }
    kill(record.pid, SIGTERM);
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(_killGracePeriod * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        if (self->_run == record) {
            kill(record.pid, SIGKILL);
        }
    });
}

- (void)cancelRun
{
    if (_run != nil) {
        [self _terminate:_run reason:@"cancelled"];
    }
}

#pragma mark Exit and adoption

static NSString *AFMResultForExitCode(int code)
{
    switch (code) {
    case 2: return @"failed:usage";
    case 3: return @"not_enough_data";
    case 4: return @"deferred";
    case 5: return @"cancelled";
    case 6: return @"discarded:retracted";
    case 7: return @"failed:helper";
    case 8: return @"failed:memory";
    case 9: return @"failed:pack";
    default: return @"failed:exit";
    }
}

- (void)_helperDidExit:(AFMPersonalRun *)record status:(int)status
{
    if (_run == record) {
        _run = nil;
    }
    BOOL signaled = WIFSIGNALED(status);
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (record.generation != _generation) {
        // Started before a clear: discarded, the clear deletes its files.
        [self _finishRun:record result:@"discarded:generation" report:nil];
        return;
    }
    NSString *staging = record.stagingDirectory;
    if (signaled || code != 0) {
        NSString *fallback = record.killReason != nil ? record.killReason : (signaled ? @"failed:crash" : AFMResultForExitCode(code));
        [self _onQueue:^id {
            NSDictionary *report = AFMReadJSONFile([staging stringByAppendingPathComponent:@"report.json"]);
            AFMRemoveItem(staging);
            return report;
        } then:^(NSDictionary *report) {
            NSString *result = fallback;
            // The helper's own result is more precise (failed:<phase>) unless
            // we killed it or it crashed.
            if (record.killReason == nil && !signaled && AFMStringValue(report[@"result"]).length > 0) {
                result = report[@"result"];
            }
            [self _finishRun:record result:result report:report];
        } discarded:^{
            [self _finishRun:record result:@"discarded:generation" report:nil];
        }];
        return;
    }
    // Exit 0: the gate was evaluated; adopt if the result says so (design D13).
    NSInteger version = MAX(MAX(self.activeVersion.integerValue, self.previousVersion.integerValue), [AFMNumberValue(_manifest[@"last_version"]) integerValue]) + 1;
    NSString *versionDirectory = [self _versionDirectory:version];
    NSString *pool = [_memoryDirectory stringByAppendingPathComponent:AFMLearningSentencePoolFileName];
    NSString *modelDirectory = _modelDirectory;
    uint64_t generation = record.generation;
    NSString *run = record.runIdentifier;
    [self _onQueue:^id {
        NSMutableDictionary *outcome = [NSMutableDictionary dictionary];
        NSDictionary *report = AFMReadJSONFile([staging stringByAppendingPathComponent:@"report.json"]);
        if (report != nil) {
            outcome[@"report"] = report;
        }
        NSString *result = AFMStringValue(report[@"result"]);
        if (report == nil || result == nil) {
            outcome[@"result"] = @"failed:report";
        } else if (![result isEqualToString:@"adopted"]) {
            outcome[@"result"] = result;
        } else if ([AFMNumberValue(report[@"generation"]) unsignedLongLongValue] != generation || ![AFMStringValue(report[@"run"]) isEqualToString:run]) {
            outcome[@"result"] = @"discarded:generation";
        } else if ([self _lateRetractionIn:pool offset:[AFMNumberValue(report[@"scan_offset"]) unsignedLongLongValue]]) {
            outcome[@"result"] = @"discarded:retracted";
        } else if (NSString *why = [self _verifyStaging:staging run:run report:report modelDirectory:modelDirectory]) {
            os_log_error(AFMPersonalLog(), "personalized version not adopted: %{public}@", why);
            outcome[@"result"] = @"failed:adopt";
        } else {
            AFMRemoveItem(versionDirectory);
            if (rename(staging.fileSystemRepresentation, versionDirectory.fileSystemRepresentation) != 0) {
                outcome[@"result"] = @"failed:adopt";
            } else {
                chmod(versionDirectory.fileSystemRepresentation, 0700);
                outcome[@"result"] = @"adopted";
                outcome[@"version"] = @(version);
            }
        }
        AFMRemoveItem(staging);
        return outcome;
    } then:^(NSDictionary *outcome) {
        NSString *result = (outcome[@"result"] != nil ? outcome[@"result"] : @"failed:adopt");
        if ([result isEqualToString:@"adopted"]) {
            [self _adoptVersion:version record:record report:outcome[@"report"]];
        }
        [self _finishRun:record result:result report:outcome[@"report"]];
    } discarded:^{
        // Cleared while adopting: nothing is published; the clear deletes
        // the staged / renamed files after this block (store queue order).
        [self _finishRun:record result:@"discarded:generation" report:nil];
    }];
}

// Store queue. Retractions written after the helper read the pool that mark
// a record with eligible segments before scan_offset (all of which are used:
// a retraction is at most 10 s old, so its record is among the newest =
// personal test set) -> discard (design D7).
- (BOOL)_lateRetractionIn:(NSString *)pool offset:(unsigned long long)offset
{
    std::unordered_set<int64_t> retracted;
    std::string error;
    if (!pz::ScanRetractions(pool.fileSystemRepresentation, offset, &retracted, &error) || retracted.empty()) {
        return NO;
    }
    if (![self _loadVocabIfNeeded]) {
        return YES;  // Cannot tell which records were used: be safe.
    }
    NSFileHandle *handle = [NSFileHandle fileHandleForReadingAtPath:pool];
    NSData *head = [handle readDataUpToLength:(NSUInteger)offset error:nil];
    [handle closeFile];
    pz::PoolScan scan;
    pz::ScanPoolBytes(std::string_view(static_cast<const char *>(head.bytes), head.length), _vocab, pz::SegmentOptions(), &scan);
    for (const auto &segment : scan.segments) {
        if (retracted.count(segment.tMs) != 0) {
            return YES;
        }
    }
    return NO;
}

// Store queue. nil when the staged files are what the gate measured and fit
// the installed factory model.
- (NSString *)_verifyStaging:(NSString *)staging run:(NSString *)run report:(NSDictionary *)report modelDirectory:(NSString *)modelDirectory
{
    pz::WeightsReadOptions options;
    std::string error;
    auto weights = pz::ReadWeightsFile([staging stringByAppendingPathComponent:@"weights.bin"].fileSystemRepresentation, options, &error);
    if (weights == nullptr) {
        return @"weights.bin invalid";
    }
    NSString *fileSHA = @(pz::ToHex(weights->factorySha).c_str());
    NSString *factory = AFMFactoryModelSHA256(modelDirectory);
    NSDictionary *config = [report[@"eval_config"] isKindOfClass:NSDictionary.class] ? report[@"eval_config"] : nil;
    if (factory == nil || ![fileSHA isEqualToString:factory] || ![AFMStringValue(config[@"factory_sha256"]) isEqualToString:factory]) {
        return @"factory model mismatch";
    }
    NSString *reportSHA = AFMStringValue(report[@"weights_sha256"]);
    if (reportSHA != nil) {
        pz::Sha256Digest digest;
        if (!pz::Sha256File([staging stringByAppendingPathComponent:@"weights.bin"].fileSystemRepresentation, &digest, &error) || ![@(pz::ToHex(digest).c_str()) isEqualToString:reportSHA]) {
            return @"weights.bin differs from the gated file";
        }
    }
    if (![AFMPreciseOverlayRun([staging stringByAppendingPathComponent:@"precise-r.conf"]) isEqualToString:run]) {
        return @"precise-r.conf missing";
    }
    return nil;
}

- (void)_adoptVersion:(NSInteger)version record:(AFMPersonalRun *)record report:(NSDictionary *)report
{
    NSNumber *active = self.activeVersion;
    NSNumber *previous = self.previousVersion;
    if (active != nil) {
        _manifest[@"previous"] = active;
    } else if (previous != nil) {
        // After 「改回出廠模型」 the old active stays as previous.
        _manifest[@"previous"] = previous;
    }
    _manifest[@"active"] = @(version);
    _manifest[@"last_version"] = @(version);
    [_manifest removeObjectForKey:@"reverted_previous"];
    NSMutableDictionary *versions = [_manifest[@"versions"] isKindOfClass:NSDictionary.class] ? [_manifest[@"versions"] mutableCopy] : [NSMutableDictionary dictionary];
    NSDictionary *config = [report[@"eval_config"] isKindOfClass:NSDictionary.class] ? report[@"eval_config"] : record.evalConfig;
    NSDictionary *gate = [report[@"gate"] isKindOfClass:NSDictionary.class] ? report[@"gate"] : @{};
    versions[[NSString stringWithFormat:@"%ld", (long)version]] = @{
        @"run" : record.runIdentifier,
        @"created" : @([self _now]),
        @"eval_config" : (config != nil ? config : @{}),
        @"gate" : gate,
    };
    _manifest[@"versions"] = versions;
    [self _writeManifest];
    [self _deleteUnreferencedVersions];
    os_log(AFMPersonalLog(), "personalized version %ld adopted; used from the next activation", (long)version);
}

- (void)_finishRun:(AFMPersonalRun *)record result:(NSString *)result report:(NSDictionary *)report
{
    [self _protectRun:record.runIdentifier protect:NO];
    if (record.generation == _generation) {
        NSMutableDictionary *last = [NSMutableDictionary dictionary];
        last[@"run"] = record.runIdentifier;
        last[@"time"] = @([self _now]);
        last[@"mode"] = record.automatic ? @"auto" : @"manual";
        last[@"result"] = result;
        NSDictionary *gate = [report[@"gate"] isKindOfClass:NSDictionary.class] ? report[@"gate"] : nil;
        for (NSString *key in @[ @"delta_personal_pt", @"delta_general_pt", @"p", @"factory_personal_pct", @"candidate_personal_pct", @"factory_general_pct", @"candidate_general_pct", @"n_personal" ]) {
            if (AFMNumberValue(gate[key]) != nil) {
                last[key] = gate[key];
            }
        }
        if (AFMNumberValue(report[@"eligible"]) != nil) {
            last[@"eligible"] = report[@"eligible"];
        }
        _manifest[@"last_run"] = last;
        // Three consecutive failures (not rejections) pause automatic runs
        // until 「立即學習」 (design D15).
        if ([result hasPrefix:@"failed:"]) {
            NSInteger failures = self.consecutiveFailures + 1;
            _manifest[@"failures"] = @(failures);
            if (failures >= _failurePauseCount) {
                _manifest[@"failure_paused"] = @YES;
            }
        } else if ([result isEqualToString:@"adopted"] || [result isEqualToString:@"not_significant"] || [result isEqualToString:@"too_small_gain"] || [result isEqualToString:@"general_regression"] || [result isEqualToString:@"worse_than_active"]) {
            _manifest[@"failures"] = @0;
        }
        [self _writeManifest];
        // One line per run: phase / reason code only, never text (design D15).
        os_log(AFMPersonalLog(), "personalization run finished: %{public}@", result);
    }
    [self _notify];
    if (_runDidFinishHandler != nil) {
        _runDidFinishHandler(result);
    }
    [self _resumeClearIfWaiting];
}

#pragma mark Versions

- (void)revertToFactory
{
    NSNumber *active = self.activeVersion;
    if (active != nil) {
        NSNumber *previous = self.previousVersion;
        if (previous != nil) {
            _manifest[@"reverted_previous"] = previous;
        }
        _manifest[@"previous"] = active;
        [_manifest removeObjectForKey:@"active"];
    }
    _manifest[@"paused"] = @YES;
    [self _writeManifest];
    [self _deleteUnreferencedVersions];
    [self _notify];
}

- (void)usePreviousVersion
{
    NSNumber *previous = self.previousVersion;
    if (previous == nil) {
        return;
    }
    NSNumber *active = self.activeVersion;
    _manifest[@"active"] = previous;
    if (active != nil) {
        _manifest[@"previous"] = active;
    } else if (AFMNumberValue(_manifest[@"reverted_previous"]) != nil) {
        // After 「改回出廠模型」: the version before becomes previous again.
        _manifest[@"previous"] = _manifest[@"reverted_previous"];
    } else {
        [_manifest removeObjectForKey:@"previous"];
    }
    [_manifest removeObjectForKey:@"reverted_previous"];
    [self _writeManifest];
    [self _notify];
}

#pragma mark Clearing (design D13)

- (void)clearAllWithCompletion:(dispatch_block_t)completion
{
    // (1) bump the generation: every queued block, adoption, load request
    // and load completion started before is discarded.
    ++_generation;
    _queueGeneration.store(_generation);
    _clearing = YES;
    if (completion != nil) {
        [_clearWaiters addObject:[completion copy]];
    }
    _manifest = [NSMutableDictionary dictionary];
    _eligibleCount = 0;
    _scannedPoolSize = 0;
    // (2) factory weights from the next activation; a personalized scorer in
    // use is replaced in the background right away.
    [AFMNeuralSelector.sharedSelector requestFactoryWeights];
    // (3) stop a running helper; (4) delete after its exit.
    if (_run != nil) {
        [self _terminate:_run reason:@"cancelled"];
    } else {
        [self _resumeClearIfWaiting];
    }
    [self _notify];
}

- (void)_resumeClearIfWaiting
{
    if (!_clearing || _run != nil) {
        return;
    }
    NSString *personal = self.personalDirectory;
    uint64_t generation = _generation;
    ++_pendingMainWork;
    dispatch_async([self _ioQueue], ^{
        AFMRemoveItem(personal);
        dispatch_async(dispatch_get_main_queue(), ^{
            --self->_pendingMainWork;
            if (self->_generation != generation) {
                return;  // Another clear took over; it finishes the waiters.
            }
            self->_clearing = NO;
            NSArray<dispatch_block_t> *waiters = [self->_clearWaiters copy];
            [self->_clearWaiters removeAllObjects];
            [self _notify];
            for (dispatch_block_t waiter in waiters) {
                waiter();
            }
        });
    });
}

#pragma mark Status

- (void)readStatusWithCompletion:(void (^)(NSDictionary *))completion
{
    NSString *path = [self.personalDirectory stringByAppendingPathComponent:@"status.json"];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        NSDictionary *status = AFMReadJSONFile(path);
        dispatch_async(dispatch_get_main_queue(), ^{
            completion(status);
        });
    });
}

#pragma mark Scheduler

- (void)startScheduler
{
    if (_scheduler != nil) {
        return;
    }
    NSString *identifier = [AFMPersonalBundleIdentifier() stringByAppendingString:@".personal-finetune"];
    _scheduler = [[NSBackgroundActivityScheduler alloc] initWithIdentifier:identifier];
    _scheduler.repeats = YES;
    _scheduler.interval = 24 * 3600;
    _scheduler.tolerance = 6 * 3600;
    _scheduler.qualityOfService = NSQualityOfServiceBackground;
    __weak AFMPersonalTrainingController *weakSelf = self;
    [_scheduler scheduleWithBlock:^(NSBackgroundActivityCompletionHandler completion) {
        dispatch_async(dispatch_get_main_queue(), ^{
            AFMPersonalTrainingController *strongSelf = weakSelf;
            if (strongSelf == nil) {
                completion(NSBackgroundActivityResultFinished);
                return;
            }
            [strongSelf handleScheduledActivity:completion];
        });
    }];
}

- (void)stopScheduler
{
    [_scheduler invalidate];
    _scheduler = nil;
}

- (void)handleScheduledActivity:(void (^)(NSBackgroundActivityResult))completion
{
    // Rescan first so the count includes today's commits.
    [self refreshWithCompletion:^{
        completion([self runAutomaticAttemptIfPossible] ? NSBackgroundActivityResultFinished : NSBackgroundActivityResultDeferred);
    }];
}

#pragma mark Testing

- (void)waitUntilIdleForTesting
{
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:30];
    do {
        dispatch_sync([self _ioQueue], ^ {});
        [NSRunLoop.currentRunLoop runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
    } while (_pendingMainWork > 0 && [deadline timeIntervalSinceNow] > 0);
}

@end
