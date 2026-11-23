import AppKit
import SwiftUI
import XCTest
@testable import RAD

@MainActor
final class WorkspaceHeaderTests: XCTestCase {
    func testCenteredTitleRemainsEditableBesideWindowControlsAndActionsAfterResize() async throws {
        guard !NSScreen.screens.isEmpty else { throw XCTSkip("The workspace header requires a window server.") }
        let app = NSApplication.shared
        let previousPolicy = app.activationPolicy()
        _ = app.setActivationPolicy(.regular)
        defer { _ = app.setActivationPolicy(previousPolicy) }
        app.finishLaunching()

        let name = String(repeating: "Enclosure with a long descriptive name ", count: 5)
            .trimmingCharacters(in: .whitespaces)
        let state = AppState()
        state.newProject()
        try state.renameProject(to: name)
        let host = NSHostingView(rootView: WorkspaceView(state: state))
        host.sizingOptions = []
        let window = NSWindow(contentRect: NSRect(x: 80, y: 80, width: 1400, height: 800),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                              backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.contentView = host
        app.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        defer {
            window.orderOut(nil)
            window.contentView = nil
            window.close()
            state.cancel()
        }

        let buttonTypes: [NSWindow.ButtonType] = [.closeButton, .miniaturizeButton, .zoomButton]
        for width in [1400, 1080] {
            window.setContentSize(NSSize(width: width, height: 800))
            let buttons = try buttonTypes.map { try XCTUnwrap(window.standardWindowButton($0)) }
            let frameView = try XCTUnwrap(host.superview)
            let headerY = frameView.bounds.maxY - WindowHeaderChrome.height / 2
            try await waitFor("Window controls should align with the header after resizing.") {
                host.layoutSubtreeIfNeeded()
                return buttons.allSatisfy { abs(frameView.convert($0.bounds, from: $0).midY - headerY) < 1 }
            }
            for button in buttons {
                let frame = frameView.convert(button.bounds, from: button)
                XCTAssertLessThanOrEqual(frame.maxX, WindowHeaderChrome.windowButtonsWidth)
                let hit = try XCTUnwrap(frameView.hitTest(NSPoint(x: frame.midX, y: frame.midY)))
                XCTAssertTrue(hit === button || hit.isDescendant(of: button))
            }
        }

        let center = NSPoint(x: host.bounds.midX,
                             y: host.isFlipped ? WindowHeaderChrome.height / 2 : host.bounds.maxY - WindowHeaderChrome.height / 2)
        try click(host.convert(center, to: nil), in: window)
        func nameField() -> NSTextField? {
            descendants(host).compactMap { $0 as? NSTextField }.first {
                $0.isEditable && host.convert($0.bounds, from: $0).contains(center)
            }
        }
        try await waitFor("Clicking the centered title should focus its inline editor.") { nameField()?.currentEditor() != nil }
        let field = try XCTUnwrap(nameField())
        XCTAssertEqual(field.stringValue, name)
        XCTAssertEqual(field.alignment, .center)
        let fieldFrame = host.convert(field.bounds, from: field)
        XCTAssertEqual(fieldFrame.midX, host.bounds.midX, accuracy: 1)
        XCTAssertGreaterThan(fieldFrame.minX, WindowHeaderChrome.windowButtonsWidth)
        // Use the leading edge of the action group, regardless of which button is first.
        let actionFrames = descendants(host).compactMap { $0 as? NSButton }.map { host.convert($0.bounds, from: $0) }
            .filter { $0.minX > center.x && $0.minY <= center.y && center.y <= $0.maxY }
        let firstActionX = try XCTUnwrap(actionFrames.map(\.minX).min())
        XCTAssertLessThan(fieldFrame.maxX, firstActionX)

        let editor = try XCTUnwrap(field.currentEditor() as? NSTextView)
        editor.selectAll(nil)
        editor.insertText("Renamed enclosure", replacementRange: editor.selectedRange())
        let enter = try XCTUnwrap(NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [],
            timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber, context: nil,
            characters: "\r", charactersIgnoringModifiers: "\r", isARepeat: false, keyCode: 36))
        app.sendEvent(enter)
        try await waitFor("Return should apply the inline rename.") { state.projectDisplayName == "Renamed enclosure" }
    }

    private func descendants(_ view: NSView) -> [NSView] { view.subviews + view.subviews.flatMap(descendants) }

    private func waitFor(_ message: String, _ predicate: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(5)
        while !predicate(), ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertTrue(predicate(), message)
    }

    private func click(_ location: NSPoint, in window: NSWindow) throws {
        func event(_ type: NSEvent.EventType) throws -> NSEvent {
            try XCTUnwrap(NSEvent.mouseEvent(with: type, location: location, modifierFlags: [],
                timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
                context: nil, eventNumber: 1, clickCount: 1, pressure: type == .leftMouseDown ? 1 : 0))
        }
        // An AppKit button can track the release before its mouse-down handler returns.
        NSApp.postEvent(try event(.leftMouseUp), atStart: true)
        NSApp.sendEvent(try event(.leftMouseDown))
        if let up = NSApp.nextEvent(matching: .leftMouseUp, until: Date(), inMode: .default, dequeue: true) {
            NSApp.sendEvent(up)
        }
    }
}
