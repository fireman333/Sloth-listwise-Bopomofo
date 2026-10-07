// Copyright (c) 2022 and onwards The McBopomofo Authors.
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

#import <Foundation/Foundation.h>
#import "KeyHandler.h"

NS_ASSUME_NONNULL_BEGIN

@interface LanguageModelManager : NSObject

+ (void)loadDataModel:(InputMode)mode;
+ (void)loadUserPhrasesWithPlainBopomofoEnabled:(BOOL)userPhraseForPlainBopomofo NS_SWIFT_NAME(loadUserPhrases(enableForPlainBopomofo:));
+ (void)loadUserPhraseReplacement;
+ (void)setupDataModelValueConverter;
+ (BOOL)checkIfUserLanguageModelFilesExist;

+ (BOOL)checkIfUserPhraseExist:(NSString *)userPhrase key:(NSString *)key NS_SWIFT_NAME(checkIfExist(userPhrase:key:));
+ (BOOL)writeUserPhrase:(NSString *)userPhrase;
+ (BOOL)removeUserPhrase:(NSString *)userPhrase;

+ (nullable NSString *)readingFor:(NSString *)phrase;

/// Reading policy (design D4): unless accepted, built-in entries and
/// associated phrases that read 癌 as ㄧㄢˊ are filtered out of both language
/// models. Bumps reloadGeneration when the policy changes.
+ (void)applyReadingPolicyAcceptingYanReadingForAi:(BOOL)accept NS_SWIFT_NAME(applyReadingPolicy(acceptYanReadingForAi:));

/// Incremented on every language-model (re)load or policy change, so that a
/// pending background re-ranking request made before it is discarded.
@property (class, readonly, nonatomic) NSUInteger reloadGeneration;

/// Personal memory (design D2): with learning enabled, the user override
/// model is the long-term one (5,000 contexts, 7-day half-life, latest pick
/// wins, refresh on commit), loaded from the memory folder; otherwise the
/// stock session-only model (500, 1.5 h). Compares the actual current
/// configuration, so it is a cheap no-op when nothing changed. Main thread;
/// called at launch and on every activation. Also turns the learning store's
/// recording on (enabled) or off (disabled, with its one final save).
+ (void)configureLearningEnabled:(BOOL)enabled NS_SWIFT_NAME(configureLearning(enabled:));
/// Whether the active user override model is the long-term configuration.
@property (class, readonly, nonatomic) BOOL learningConfigurationActive;
/// Clears all learning data (design D11): deletes the memory file, files set
/// aside as unreadable and the sentence pool, and empties the in-memory model
/// (keeping its configuration). Main thread.
+ (void)clearLearningData;
/// Saves pending memory changes now (deactivateServer).
+ (void)flushLearningData;

@property (class, readonly, nonatomic) NSString *dataFolderPath;
@property (class, readonly, nonatomic) NSString *userPhrasesDataPathMcBopomofo;
@property (class, readonly, nonatomic) NSString *userPhrasesDataPathPlainBopomofo;
@property (class, readonly, nonatomic) NSString *excludedPhrasesDataPathMcBopomofo;
@property (class, readonly, nonatomic) NSString *excludedPhrasesDataPathPlainBopomofo;
@property (class, readonly, nonatomic) NSString *phraseReplacementDataPathMcBopomofo;
@property (class, assign, nonatomic) BOOL phraseReplacementEnabled;

@end

/// The following methods are merely for testing.
@interface LanguageModelManager ()
+ (void)loadDataModels;
/// Testing only: forgets every observation of the user override model
/// (keeps its configuration: stock or long-term).
+ (void)resetUserOverrideModelForTesting;
/// Testing only: the stock model without saving anything, as at process
/// start (the next configureLearning(enabled: true) loads the memory file).
+ (void)resetLearningConfigurationForTesting;
/// Testing only: number of contexts in the user override model.
@property (class, readonly, nonatomic) NSUInteger userOverrideModelSizeForTesting;
/// Testing only: empties the Bopomofo-mode user and excluded phrases.
+ (void)clearUserPhrasesForTesting;
/// Testing only: replaces the Bopomofo-mode user phrases with content
/// (user-phrase file format) and empties the excluded phrases.
+ (void)loadUserPhrasesForTesting:(NSString *)content;
@end

@interface LanguageModelManager ()
+ (NSString *)annotateVariantForCharacters:(NSString *)characters readings:(NSString *)readings NS_SWIFT_NAME(annotateVariant(characters:readings:));
@end

NS_ASSUME_NONNULL_END
