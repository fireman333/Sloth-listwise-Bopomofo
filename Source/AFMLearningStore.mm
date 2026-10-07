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

#import "AFMLearningStore.h"
#import "LanguageModelManager+Privates.h"
#import "LanguageModelManager.h"
#import "McBopomofo-Swift.h"
#import "UserOverrideModel.h"

#import <Carbon/Carbon.h>
#import <os/log.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

NSString *const AFMLearningMemoryFileName = @"override-model.txt";
NSString *const AFMLearningSentencePoolFileName = @"sentences.jsonl";
static NSString *const kAFMBadMemoryFilePrefix = @"override-model.bad-";
static const NSTimeInterval kAFMDefaultSaveDelay = 3.0;

static os_log_t AFMLearningLog(void)
{
    static os_log_t log;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSString *identifier = NSBundle.mainBundle.bundleIdentifier;
        if (identifier == nil) {
            identifier = @"io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo";
        }
        log = os_log_create(identifier.UTF8String, "Learning");
    });
    return log;
}

// Write errors are logged once per process (design D4).
static void AFMLogWriteErrorOnce(const char *what, int error)
{
    static std::atomic<bool> logged { false };
    bool expected = false;
    if (logged.compare_exchange_strong(expected, true)) {
        os_log_error(AFMLearningLog(), "learning data %{public}s failed: errno %d", what, error);
    }
}

// Store queue only.
static bool AFMEnsureDirectory(NSString *directory)
{
    BOOL isDirectory = NO;
    if ([NSFileManager.defaultManager fileExistsAtPath:directory isDirectory:&isDirectory] && isDirectory) {
        return true;
    }
    NSError *error = nil;
    BOOL ok = [NSFileManager.defaultManager createDirectoryAtPath:directory
                                      withIntermediateDirectories:YES
                                                       attributes:@{ NSFilePosixPermissions : @(0700) }
                                                            error:&error];
    if (!ok) {
        AFMLogWriteErrorOnce("mkdir", (int)error.code);
        return false;
    }
    chmod(directory.fileSystemRepresentation, 0700);
    return true;
}

static bool AFMWriteAll(int fd, const std::string& data)
{
    const char *p = data.data();
    size_t left = data.size();
    while (left > 0) {
        ssize_t n = write(fd, p, left);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        p += n;
        left -= static_cast<size_t>(n);
    }
    return true;
}

// Store queue only: .tmp (0600) then rename.
static void AFMWriteFileAtomically(NSString *directory, NSString *path, const std::string& data)
{
    if (!AFMEnsureDirectory(directory)) {
        return;
    }
    std::string target = path.fileSystemRepresentation;
    std::string tmp = target + ".tmp";
    int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
        AFMLogWriteErrorOnce("open", errno);
        return;
    }
    fchmod(fd, 0600);
    bool ok = AFMWriteAll(fd, data);
    int writeError = errno;
    close(fd);
    if (!ok) {
        AFMLogWriteErrorOnce("write", writeError);
        unlink(tmp.c_str());
        return;
    }
    if (rename(tmp.c_str(), target.c_str()) != 0) {
        AFMLogWriteErrorOnce("rename", errno);
        unlink(tmp.c_str());
    }
}

// Store queue only.
static void AFMAppendLine(NSString *directory, NSString *path, const std::string& line)
{
    if (!AFMEnsureDirectory(directory)) {
        return;
    }
    int fd = open(path.fileSystemRepresentation, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) {
        AFMLogWriteErrorOnce("open", errno);
        return;
    }
    if (!AFMWriteAll(fd, line)) {
        AFMLogWriteErrorOnce("append", errno);
    }
    close(fd);
}

static void AFMAppendJSONString(std::string& out, const std::string& s)
{
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    out += '"';
}

// {"v":1,"t":<epoch>,"manual":<bool>,"walk":"...","app":"...","model":"...",
//  "nodes":[["<reading>","<raw>","<src>"],...]} + "\n" (design D4).
static std::string AFMEncodeRecord(const AFMSentenceRecord& record, const std::string& appVersion)
{
    std::string out = "{\"v\":1,\"t\":";
    char number[64];
    snprintf(number, sizeof(number), "%.3f", record.timestamp);
    out += number;
    out += ",\"manual\":";
    out += record.manual ? "true" : "false";
    out += ",\"walk\":";
    AFMAppendJSONString(out, record.walk);
    out += ",\"app\":";
    AFMAppendJSONString(out, appVersion);
    out += ",\"model\":";
    AFMAppendJSONString(out, record.modelIdentifier);
    out += ",\"nodes\":[";
    bool first = true;
    for (const auto& node : record.nodes) {
        if (!first) {
            out += ',';
        }
        first = false;
        out += '[';
        AFMAppendJSONString(out, node.reading);
        out += ',';
        AFMAppendJSONString(out, node.raw);
        out += ',';
        AFMAppendJSONString(out, std::string(1, node.source));
        out += ']';
    }
    out += "]}\n";
    return out;
}

static double AFMSystemClock(void)
{
    return NSDate.date.timeIntervalSince1970;
}

// Secure input counts only when the client application itself holds it (a
// password field, or Terminal's Secure Keyboard Entry). A system-wide check is
// not usable: loginwindow often keeps secure input enabled after the screen
// was locked, which would pause learning indefinitely. An unknown client
// (pid 0) is treated as not secure; macOS already routes real password fields
// away from input methods.
BOOL AFMSecureInputHeldByClient(BOOL secureInputEnabled, int holderPID, pid_t clientPID)
{
    return secureInputEnabled && clientPID > 0 && holderPID > 0 && holderPID == clientPID;
}

static BOOL AFMSystemSecureInputCheck(pid_t clientPID)
{
    if (clientPID <= 0 || !IsSecureEventInputEnabled()) {
        return NO;
    }
    CFDictionaryRef session = CGSessionCopyCurrentDictionary();
    if (session == NULL) {
        return NO;
    }
    CFNumberRef holder = (CFNumberRef)CFDictionaryGetValue(session, CFSTR("kCGSSessionSecureInputPID"));
    int holderPID = 0;
    if (holder == NULL || CFGetTypeID(holder) != CFNumberGetTypeID() || !CFNumberGetValue(holder, kCFNumberIntType, &holderPID)) {
        holderPID = 0;
    }
    CFRelease(session);
    return AFMSecureInputHeldByClient(YES, holderPID, clientPID);
}

@implementation AFMLearningStore {
    dispatch_queue_t _queue;
    // Bumped only by clearAllData; queued work captured before a clear no-ops.
    std::atomic<uint64_t> _generation;
    NSTimer *_saveTimer;
    BOOL _dirty;
    BOOL _lastSecureInput;
    std::string _appVersion;
}

@synthesize recordingEnabled = _recordingEnabled;
@synthesize directory = _directory;
@synthesize saveDelay = _saveDelay;
@synthesize clock = _clock;
@synthesize secureInputCheck = _secureInputCheck;
@synthesize clientProcessIdentifier = _clientProcessIdentifier;
@synthesize lastLoadMilliseconds = _lastLoadMilliseconds;
@synthesize lastSerializeMilliseconds = _lastSerializeMilliseconds;
@synthesize saveCountForTesting = _saveCountForTesting;

+ (AFMLearningStore *)sharedStore
{
    static AFMLearningStore *store;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        store = [[AFMLearningStore alloc] init];
    });
    return store;
}

- (instancetype)init
{
    self = [super init];
    if (self) {
        NSString *identifier = NSBundle.mainBundle.bundleIdentifier;
        if (identifier == nil) {
            identifier = @"McBopomofo";
        }
        NSString *label = [identifier stringByAppendingString:@".Learning"];
        _queue = dispatch_queue_create(label.UTF8String, DISPATCH_QUEUE_SERIAL);
        _generation = 0;
        _directory = [UserPhraseLocationHelper.learningDataDirectory copy];
        _saveDelay = kAFMDefaultSaveDelay;
        NSDictionary *info = NSBundle.mainBundle.infoDictionary;
        NSString *version = info[@"CFBundleShortVersionString"];
        NSString *build = info[@"CFBundleVersion"];
        _appVersion = std::string([NSString stringWithFormat:@"%@ (%@)", version != nil ? version : @"", build != nil ? build : @""].UTF8String);
    }
    return self;
}

#pragma mark - Paths

- (NSString *)memoryFilePath
{
    return [_directory stringByAppendingPathComponent:AFMLearningMemoryFileName];
}

- (NSString *)sentencePoolFilePath
{
    return [_directory stringByAppendingPathComponent:AFMLearningSentencePoolFileName];
}

#pragma mark - Clock and secure input

- (double)now
{
    return _clock ? _clock() : AFMSystemClock();
}

- (BOOL)isSecureInputEnabled
{
    BOOL secure = _secureInputCheck ? _secureInputCheck() : AFMSystemSecureInputCheck(_clientProcessIdentifier);
    if (secure && !_lastSecureInput) {
        // No text content, only the transition (design D6).
        os_log(AFMLearningLog(), "learning paused: the client holds secure event input");
    }
    _lastSecureInput = secure;
    return secure;
}

#pragma mark - Recording and saving

- (void)setRecordingEnabled:(BOOL)enabled
{
    if (enabled) {
        _recordingEnabled = YES;
        return;
    }
    [self _cancelTimer];
    if (_recordingEnabled && _dirty && LanguageModelManager.learningConfigurationActive) {
        // The one final save at the moment learning is turned off (design D2).
        [self _writeMemorySynchronously:YES];
    }
    _dirty = NO;
    _recordingEnabled = NO;
}

- (BOOL)hasUnsavedChanges
{
    return _dirty;
}

- (void)markDirty
{
    if (!_recordingEnabled) {
        return;
    }
    _dirty = YES;
    if (_saveTimer != nil) {
        return;
    }
    __weak AFMLearningStore *weakSelf = self;
    _saveTimer = [NSTimer timerWithTimeInterval:_saveDelay repeats:NO block:^(NSTimer *timer) {
        AFMLearningStore *strongSelf = weakSelf;
        if (strongSelf == nil || strongSelf->_saveTimer != timer) {
            return;
        }
        strongSelf->_saveTimer = nil;
        [strongSelf _saveIfNeeded];
    }];
    [NSRunLoop.mainRunLoop addTimer:_saveTimer forMode:NSRunLoopCommonModes];
}

- (void)flush
{
    [self _cancelTimer];
    [self _saveIfNeeded];
}

- (void)_cancelTimer
{
    [_saveTimer invalidate];
    _saveTimer = nil;
}

- (std::string)_serializeActiveModel
{
    auto t0 = std::chrono::steady_clock::now();
    std::ostringstream os;
    LanguageModelManager.userOverrideModel->serialize(os);
    std::string data = os.str();
    _lastSerializeMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return data;
}

- (void)_saveIfNeeded
{
    if (!_dirty || !_recordingEnabled) {
        return;
    }
    _dirty = NO;
    // A stock (session-only) model never overwrites the saved memory
    // (review R2-8).
    if (!LanguageModelManager.learningConfigurationActive) {
        return;
    }
    [self _writeMemorySynchronously:NO];
}

/// Serializes the active model now and writes it on the store queue, unless
/// a clear bumps the generation first.
- (void)_writeMemorySynchronously:(BOOL)synchronously
{
    std::string data = [self _serializeActiveModel];
    NSString *directory = _directory;
    NSString *path = self.memoryFilePath;
    uint64_t generation = _generation.load();
    dispatch_block_t write = ^{
        if (self->_generation.load() == generation) {
            AFMWriteFileAtomically(directory, path, data);
        }
    };
    if (synchronously) {
        dispatch_sync(_queue, write);
    } else {
        dispatch_async(_queue, write);
    }
    ++_saveCountForTesting;
}

#pragma mark - Loading

- (BOOL)loadMemoryIntoModel:(McBopomofo::UserOverrideModel &)model
{
    auto t0 = std::chrono::steady_clock::now();
    NSString *path = self.memoryFilePath;
    __block std::string contents;
    __block BOOL exists = NO;
    __block BOOL readable = NO;
    // After any queued write, so the newest saved state is read.
    dispatch_sync(_queue, ^{
        exists = [NSFileManager.defaultManager fileExistsAtPath:path];
        NSData *data = exists ? [NSData dataWithContentsOfFile:path] : nil;
        if (data != nil) {
            readable = YES;
            contents.assign(static_cast<const char *>(data.bytes), data.length);
        }
    });
    BOOL ok = YES;
    if (!exists) {
        std::istringstream empty("AFM-UOM 1\n");
        model.deserialize(empty);
    } else {
        std::istringstream is(contents);
        // An existing but unreadable file is set aside like a malformed one,
        // so the next save does not overwrite it.
        if (!readable || !model.deserialize(is)) {
            if (!readable) {
                std::istringstream empty("AFM-UOM 1\n");
                model.deserialize(empty);
            }
            ok = NO;
            NSString *badName = [NSString stringWithFormat:@"%@%lld.txt", kAFMBadMemoryFilePrefix, (long long)[self now]];
            NSString *badPath = [_directory stringByAppendingPathComponent:badName];
            dispatch_sync(_queue, ^{
                if (rename(path.fileSystemRepresentation, badPath.fileSystemRepresentation) != 0) {
                    AFMLogWriteErrorOnce("set aside", errno);
                }
            });
            os_log_error(AFMLearningLog(), "malformed memory file set aside");
        }
    }
    _lastLoadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

#pragma mark - Sentence pool

- (void)appendSentenceRecord:(AFMSentenceRecord)record
{
    if (!_recordingEnabled) {
        return;
    }
    NSString *directory = _directory;
    NSString *path = self.sentencePoolFilePath;
    uint64_t generation = _generation.load();
    std::string appVersion = _appVersion;
    auto shared = std::make_shared<AFMSentenceRecord>(std::move(record));
    dispatch_async(_queue, ^{
        if (self->_generation.load() != generation) {
            return;
        }
        AFMAppendLine(directory, path, AFMEncodeRecord(*shared, appVersion));
    });
}

- (void)appendRetractionForTimestamp:(double)timestamp
{
    if (!_recordingEnabled) {
        return;
    }
    NSString *directory = _directory;
    NSString *path = self.sentencePoolFilePath;
    uint64_t generation = _generation.load();
    dispatch_async(_queue, ^{
        if (self->_generation.load() != generation) {
            return;
        }
        char line[96];
        snprintf(line, sizeof(line), "{\"v\":1,\"retract\":%.3f}\n", timestamp);
        AFMAppendLine(directory, path, std::string(line));
    });
}

- (void)performOnStoreQueue:(dispatch_block_t)block
{
    uint64_t generation = _generation.load();
    dispatch_async(_queue, ^{
        if (self->_generation.load() == generation) {
            block();
        }
    });
}

- (dispatch_queue_t)storeQueue
{
    return _queue;
}

#pragma mark - Clearing

- (void)clearAllData
{
    _generation.fetch_add(1);
    [self _cancelTimer];
    _dirty = NO;
    NSString *directory = _directory;
    dispatch_async(_queue, ^{
        NSFileManager *fm = NSFileManager.defaultManager;
        for (NSString *name in [fm contentsOfDirectoryAtPath:directory error:nil]) {
            BOOL learningFile = [name isEqualToString:AFMLearningMemoryFileName]
                || [name isEqualToString:[AFMLearningMemoryFileName stringByAppendingString:@".tmp"]]
                || [name isEqualToString:AFMLearningSentencePoolFileName]
                || ([name hasPrefix:kAFMBadMemoryFilePrefix] && [name hasSuffix:@".txt"]);
            if (learningFile) {
                NSString *path = [directory stringByAppendingPathComponent:name];
                if (unlink(path.fileSystemRepresentation) != 0 && errno != ENOENT) {
                    AFMLogWriteErrorOnce("delete", errno);
                }
            }
        }
    });
}

#pragma mark - Testing

- (void)waitUntilIdleForTesting
{
    dispatch_sync(_queue, ^ {});
}

- (void)suspendQueueForTesting
{
    dispatch_suspend(_queue);
}

- (void)resumeQueueForTesting
{
    dispatch_resume(_queue);
}

- (void)resetForTesting
{
    [self _cancelTimer];
    _dirty = NO;
    _recordingEnabled = NO;
    _directory = [UserPhraseLocationHelper.learningDataDirectory copy];
    _clock = nil;
    _secureInputCheck = nil;
    _lastSecureInput = NO;
    _saveDelay = kAFMDefaultSaveDelay;
    _saveCountForTesting = 0;
}

@end
