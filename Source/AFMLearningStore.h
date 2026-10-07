// AFMLearningStore: personal-memory storage (openspec change personal-memory,
// design D3 / D4 / D11). Persists the long-term user override model under
// <application support root>/memory/override-model.txt and appends the
// sentence pool to memory/sentences.jsonl (both mode 0600, folder 0700).
//
// Threading: every public method runs on the main thread. Files are written
// on a private serial queue; every queued block captures the store
// generation and no-ops if a clear bumped it meanwhile. Logs never contain
// typed text.

#import <Foundation/Foundation.h>

#ifdef __cplusplus
#include <string>
#include <vector>

namespace McBopomofo {
class UserOverrideModel;
}
#endif

NS_ASSUME_NONNULL_BEGIN

/// File names inside the learning data directory (design D3 / D4).
extern NSString *const AFMLearningMemoryFileName;    // override-model.txt
extern NSString *const AFMLearningSentencePoolFileName;  // sentences.jsonl

#ifdef __cplusplus
/// One node of a sentence-pool record (design D4).
struct AFMSentenceRecordNode {
    std::string reading;
    // Dictionary characters before phrase replacement, macros and Simplified
    // conversion (Unigram::rawValue(), or value() when empty).
    std::string raw;
    // p manual pick, a associated phrase, m memory suggestion, q precise
    // result, u user phrase, w walk (model / dictionary).
    char source = 'w';
};

/// A sentence-pool record before JSON encoding (encoded on the store queue).
struct AFMSentenceRecord {
    double timestamp = 0;
    bool manual = false;
    // "stock", "r" or "precise".
    std::string walk;
    std::string modelIdentifier;
    std::vector<AFMSentenceRecordNode> nodes;
};
#endif

/// Personal memory storage (design D3): loads and saves the long-term user
/// override model and appends sentence-pool records. The API is main-thread
/// only; file I/O runs on a private serial queue, except the synchronous
/// load and the one-time final save when recording is turned off.
/// Pure decision used by the system secure-input check: secure input pauses
/// learning only when it is enabled and held by the client process (both pids
/// known and equal).
FOUNDATION_EXPORT BOOL AFMSecureInputHeldByClient(BOOL secureInputEnabled, int holderPID, pid_t clientPID);

/// Private learning-data files (Codex review 2026-10-08, finding 2; spec
/// personal-memory "files readable and writable only by the user").
/// Opens `path` with `flags` (O_CREAT etc.) plus O_NOFOLLOW | O_CLOEXEC, mode
/// 0600; refuses (-1, errno ELOOP / EPERM) a symlink, a non-regular file, a
/// file owned by another user or with more than one link; an existing file
/// with wider permissions is narrowed to 0600 (fchmod). Any thread.
FOUNDATION_EXPORT int AFMOpenPrivateFile(NSString *path, int flags);
/// Creates `directory` (with intermediates, 0700) or checks the existing one:
/// must be a directory owned by the user; its mode is set to 0700. Any thread.
FOUNDATION_EXPORT BOOL AFMEnsurePrivateDirectory(NSString *directory);
/// Writes `data` to `path` via a fresh `<path>.tmp` (O_EXCL | O_NOFOLLOW,
/// 0600) and rename(2) (which replaces a symlink at `path` itself, never its
/// target). The parent directory is made private first. Any thread.
FOUNDATION_EXPORT BOOL AFMWritePrivateFileAtomically(NSString *path, NSData *data);

@interface AFMLearningStore : NSObject

@property (class, readonly, nonatomic) AFMLearningStore *sharedStore NS_SWIFT_NAME(shared);

/// Whether saves and pool records are produced. Turned on by
/// LanguageModelManager.configureLearning(enabled: true); the learning
/// preference setter calls setRecordingEnabled: immediately.
@property (readonly, nonatomic) BOOL recordingEnabled;

/// Turning recording off cancels the pending save timer and, if the active
/// user override model is the long-term one and has unsaved changes, writes
/// it synchronously once (design D2). Turning it on only sets the flag.
- (void)setRecordingEnabled:(BOOL)enabled;

/// The learning data directory. Defaults to
/// UserPhraseLocationHelper.learningDataDirectory; created lazily (0700).
@property (copy, nonatomic) NSString *directory;
@property (readonly, nonatomic) NSString *memoryFilePath;
@property (readonly, nonatomic) NSString *sentencePoolFilePath;

/// Delay between the first unsaved change and the save. Default 3 s.
@property (assign, nonatomic) NSTimeInterval saveDelay;

/// Wall-clock time (seconds since 1970) used for the user override model and
/// the pool. Injectable for tests; nil restores the system clock.
@property (copy, nonatomic, nullable) double (^clock)(void);
- (double)now;

/// Process identifier of the current client application (0 = unknown), set on
/// activation. Secure input pauses learning only when this process holds it.
@property (nonatomic) pid_t clientProcessIdentifier;

/// Secure event input check (design D6), injectable for tests; nil restores
/// the system check (secure input held by the client process). Logs once per
/// transition to enabled.
@property (copy, nonatomic, nullable) BOOL (^secureInputCheck)(void);
- (BOOL)isSecureInputEnabled;

/// Called after the long-term model changed (observe / refresh). Schedules a
/// save saveDelay seconds later unless one is already scheduled. No-op while
/// recording is off.
- (void)markDirty;
@property (readonly, nonatomic) BOOL hasUnsavedChanges;

/// Saves now if there are unsaved changes (deactivateServer).
- (void)flush;

/// Deletes the memory file, files set aside as unreadable and the sentence
/// pool (queued after earlier work, which no-ops), bumps the generation and
/// cancels the pending save. Use LanguageModelManager.clearLearningData().
- (void)clearAllData;

#ifdef __cplusplus
/// Encodes and appends one record on the store queue. No-op while recording
/// is off.
- (void)appendSentenceRecord:(AFMSentenceRecord)record;

/// Replaces the model's content with the memory file (synchronously, after
/// any queued write). A missing file leaves the model empty and returns YES.
/// A malformed file leaves it empty, is renamed to
/// override-model.bad-<epoch>.txt and returns NO.
- (BOOL)loadMemoryIntoModel:(McBopomofo::UserOverrideModel &)model;
#endif

/// Personal-finetune design D7: appends {"v":1,"retract":<t>} for the pool
/// record committed at `timestamp` on the store queue (same generation rule
/// as records). No-op while recording is off.
- (void)appendRetractionForTimestamp:(double)timestamp;

/// The store queue, for work that must be ordered with appends and clears.
@property (readonly, nonatomic) dispatch_queue_t storeQueue;

/// Timings of the last load / serialization on the main thread, in ms.
@property (readonly, nonatomic) double lastLoadMilliseconds;
@property (readonly, nonatomic) double lastSerializeMilliseconds;

/// Testing only.
- (void)waitUntilIdleForTesting;
- (void)suspendQueueForTesting;
- (void)resumeQueueForTesting;
/// Recording off without saving, no timer, default directory, clock, secure
/// check and save delay.
- (void)resetForTesting;
@property (readonly, nonatomic) NSUInteger saveCountForTesting;

@end

NS_ASSUME_NONNULL_END
