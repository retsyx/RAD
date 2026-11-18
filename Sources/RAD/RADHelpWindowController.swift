import AppKit
import SwiftUI

/// App-wide reference material has no Project state or restoration lifecycle.
@MainActor
final class RADHelpWindowController: NSWindowController {
    init() { super.init(window: nil) }

    required init?(coder: NSCoder) { nil }

    private func makeWindow() -> NSWindow {
        let size = NSSize(width: 560, height: 480)
        let help = NSWindow(contentRect: NSRect(origin: .zero, size: size),
                            styleMask: [.titled, .closable, .miniaturizable, .resizable],
                            backing: .buffered, defer: false)
        help.title = "RAD Help"
        help.identifier = NSUserInterfaceItemIdentifier("RAD.Help")
        help.isReleasedWhenClosed = false
        help.isRestorable = false
        help.tabbingMode = .disallowed

        // Keep native window movement and sizing independent of hosted layout.
        let container = NSView(frame: NSRect(origin: .zero, size: size))
        let host = NSHostingView(rootView: RADHelpView().preferredColorScheme(.dark).tint(Palette.accent))
        host.sizingOptions = []
        host.frame = container.bounds
        host.autoresizingMask = [.width, .height]
        container.addSubview(host)
        help.contentView = container
        help.contentMinSize = NSSize(width: 520, height: 420)
        help.center()
        return help
    }

    override func showWindow(_ sender: Any?) {
        if window == nil { window = makeWindow() }
        super.showWindow(sender)
        if window?.isMiniaturized == true { window?.deminiaturize(sender) }
        window?.makeKeyAndOrderFront(sender)
    }
}
