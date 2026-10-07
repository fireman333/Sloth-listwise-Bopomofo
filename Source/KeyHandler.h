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

@class KeyHandlerInput;
@class InputState;

NS_ASSUME_NONNULL_BEGIN

typedef NSString *const InputMode NS_TYPED_ENUM;
extern InputMode InputModeBopomofo;
extern InputMode InputModePlainBopomofo;

@class KeyHandler;

@protocol KeyHandlerDelegate <NSObject>
- (id)candidateControllerForKeyHandler:(KeyHandler *)keyHandler;
- (void)keyHandler:(KeyHandler *)keyHandler didSelectCandidateAtIndex:(NSInteger)index candidateController:(id)controller;
- (BOOL)keyHandler:(KeyHandler *)keyHandler didRequestWriteUserPhraseWithState:(InputState *)state;
- (BOOL)keyHandler:(KeyHandler *)keyHandler didRequestBoostScoreForPhrase:(NSString *)phrase reading:(NSString *)reading;
- (BOOL)keyHandler:(KeyHandler *)keyHandler didRequestExcludePhrase:(NSString *)phrase reading:(NSString *)reading;
- (BOOL)keyHandlerDidRequestReloadLanguageModel:(KeyHandler *)keyHandler;
@optional
/// Precise mode (design D3): whether a background result may replace the
/// composing buffer, i.e. the current state is still the plain inputting
/// state showing composingBuffer and the client is still active.
- (BOOL)keyHandler:(KeyHandler *)keyHandler canApplyPreciseResultForComposingBuffer:(NSString *)composingBuffer;
/// Precise mode: the inputting state after a background result was applied.
- (void)keyHandler:(KeyHandler *)keyHandler didUpdatePreciseState:(InputState *)state;
@end

@interface BuildAssociatedPhraseParams: NSObject
@property (strong, nonatomic) id previousState;
@property (assign, nonatomic) size_t prefixCursorIndex;
@property (strong, nonatomic) NSString *reading;
@property (strong, nonatomic) NSString *value;
@property (assign, nonatomic) NSInteger candidateIndex;
@property (assign, nonatomic) BOOL useVerticalMode;
@property (assign, nonatomic) BOOL autoTriggered;
@end

@interface KeyHandler : NSObject

- (BOOL)handleInput:(KeyHandlerInput *)input
              state:(InputState *)state
      stateCallback:(void (^)(InputState *))stateCallback
      errorCallback:(void (^)(void))errorCallback NS_SWIFT_NAME(handle(input:state:stateCallback:errorCallback:));
- (BOOL)handleAssociatedPhraseWithState:(InputState *)state
                        useVerticalMode:(BOOL)useVerticalMode
                          stateCallback:(void (^)(InputState *))stateCallback
                          errorCallback:(void (^)(void))errorCallback
                            autoTriggered:(BOOL)useShiftKey
                      maxCandidateCount:(size_t)maxCandidateCount;

- (void)syncWithPreferences;
- (void)fixNodeWithReading:(NSString *)reading
                                 value:(NSString *)value
                   originalCursorIndex:(size_t)originalCursorIndex
    useMoveCursorAfterSelectionSetting:(BOOL)flag NS_SWIFT_NAME(fixNode(reading:value:originalCursorIndex:useMoveCursorAfterSelectionSetting:));
- (void)fixNodeForAssociatedPhraseWithPrefixAt:(size_t)prefixCursorIndex
                                 prefixReading:(NSString *)pfxReading
                                   prefixValue:(NSString *)pfxValue
                       associatedPhraseReading:(NSString *)phraseReading
                         associatedPhraseValue:(NSString *)phraseValue;
- (void)clear;

/// End-of-key-event hook for precise mode (design D1). Call once after each
/// key event and after a candidate pick; pass the composing buffer if the
/// resulting state is InputState.Inputting, nil otherwise. Makes at most one
/// background re-ranking request; never blocks.
- (void)requestPreciseIfNeededForComposingBuffer:(nullable NSString *)composingBuffer
    NS_SWIFT_NAME(requestPreciseIfNeeded(composingBuffer:));

- (void)handleForceCommitWithStateCallback:(void (^)(InputState *))stateCallback
    NS_SWIFT_NAME(handleForceCommit(stateCallback:));

- (InputState *)buildInputtingState;

- (nullable InputState *)buildAssociatedPhrasePlainStateWithReading:(NSString *)reading
                                                              value:(NSString *)value
                                                    useVerticalMode:(BOOL)useVerticalMode;
- (nullable InputState *)buildAssociatedPhraseStateWithParams:(BuildAssociatedPhraseParams *)params;

- (size_t)computeActualCursorIndex:(size_t)cursor;

- (NSArray<NSString *> *)collectUserFileIssues;

@property (strong, nonatomic) InputMode inputMode;
@property (weak, nonatomic) id<KeyHandlerDelegate> delegate;
@property (assign, nonatomic, readonly) NSInteger actualCandidateCursorIndex;
@property (assign, nonatomic, readonly) NSInteger cursorIndex;
/// Bumped on commit, clear, input-mode switch, candidate window, boost /
/// exclude: a precise result requested before it is never applied.
@property (assign, nonatomic, readonly) NSUInteger compositionGeneration;
/// Precise-mode counters and state, for tests.
@property (assign, nonatomic, readonly) NSUInteger preciseRequestCount;
@property (assign, nonatomic, readonly) NSUInteger preciseAppliedCount;
@property (assign, nonatomic, readonly) BOOL showingPreciseResult;
/// Wall time of the last applied precise result on the key-handling thread
/// (checks, re-walk, inputting state, delegate update), in ms.
@property (assign, nonatomic, readonly) double lastPreciseApplyMilliseconds;
/// Wall time of the last N-best + gate computation in the hook, in ms (tests).
@property (assign, nonatomic, readonly) double lastPrecisePayloadMilliseconds;
/// Test-only: re-evaluates the composing buffer without any change.
- (void)rewalkForTesting;
@end

NS_ASSUME_NONNULL_END
