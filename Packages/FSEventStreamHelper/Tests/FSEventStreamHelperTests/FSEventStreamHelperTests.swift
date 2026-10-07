// Copyright (c) 2011 and onwards The McBopomofo Authors.
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

import XCTest
@testable import FSEventStreamHelper

final class FSEventStreamHelperTests: XCTestCase {
    func testEventPathIsPublic() {
        let event = FSEventStreamHelper.Event(
            path: "/tmp/a/b.txt",
            flags: FSEventStreamEventFlags(kFSEventStreamEventFlagItemModified),
            id: 42)
        XCTAssertEqual(event.path, "/tmp/a/b.txt")
        XCTAssertEqual(event.id, 42)
    }

    func testReportsFileLevelEvents() throws {
        let fm = FileManager.default
        let root = (NSTemporaryDirectory() as NSString).appendingPathComponent(
            "FSEventStreamHelperTests-\(ProcessInfo.processInfo.processIdentifier)")
        try? fm.removeItem(atPath: root)
        try fm.createDirectory(atPath: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: root) }

        final class Collector: FSEventStreamHelperDelegate {
            let expectation: XCTestExpectation
            let fileName: String
            var paths: [String] = []
            init(expectation: XCTestExpectation, fileName: String) {
                self.expectation = expectation
                self.fileName = fileName
            }
            func helper(_ helper: FSEventStreamHelper, didReceive events: [FSEventStreamHelper.Event]) {
                paths.append(contentsOf: events.map(\.path))
                if events.contains(where: { $0.path.hasSuffix("/" + fileName) }) {
                    expectation.fulfill()
                }
            }
        }

        let helper = FSEventStreamHelper(path: root, queue: DispatchQueue(label: "FSEventStreamHelperTests"))
        let collector = Collector(
            expectation: expectation(description: "file event"), fileName: "file.txt")
        helper.delegate = collector
        XCTAssertTrue(helper.start())
        defer { helper.stop() }
        // Give the stream a moment to start before writing.
        Thread.sleep(forTimeInterval: 0.5)
        try "x".write(toFile: (root as NSString).appendingPathComponent("file.txt"), atomically: false, encoding: .utf8)
        wait(for: [collector.expectation], timeout: 10)
    }
}
