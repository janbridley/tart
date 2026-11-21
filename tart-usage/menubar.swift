import AppKit
import Darwin

private let home = URL(
  fileURLWithPath: ProcessInfo.processInfo.environment["HOME"] ?? NSHomeDirectory(),
  isDirectory: true
)
private let config = home.appending(path: ".config/tart", directoryHint: .isDirectory)
private let ledgerURL = config.appending(path: "usage.jsonl")
private let providersURL = config.appending(path: "providers.toml")
private let executable =
  Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
private let loginLabel = "local.tartusage"
private let loginAgent = home.appending(path: "Library/LaunchAgents/\(loginLabel).plist")

@main
@MainActor
final class TartUsage: NSObject, NSApplicationDelegate, NSMenuDelegate {
  private let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
  private let ledger = Ledger(url: ledgerURL)
  private let checkmark = NSImage(systemSymbolName: "checkmark", accessibilityDescription: nil)
  private var report: UsageReport?
  private var refreshing = false
  private var displayedText = ""

  func applicationDidFinishLaunching(_ notification: Notification) {
    item.button?.image = NSImage(
      systemSymbolName: "chart.pie", accessibilityDescription: "tart usage")
    item.button?.imagePosition = .imageOnly
    item.button?.toolTip = "tart usage — all sessions\n\(ledgerURL.path)"

    let menu = NSMenu()
    menu.delegate = self
    menuNeedsUpdate(menu)
    item.menu = menu
    let timer = Timer(
      timeInterval: 60, target: self, selector: #selector(refresh), userInfo: nil, repeats: true)
    RunLoop.main.add(timer, forMode: .common)
  }

  @objc private func refresh() {
    guard !refreshing else { return }
    refreshing = true
    Task {
      report = await ledger.refresh(providers: providersURL)
      refreshing = false
    }
  }

  func menuNeedsUpdate(_ menu: NSMenu) {
    refresh()
    menu.removeAllItems()
    menu.addItem(labelItem("tart usage"))
    menu.addItem(labelItem(report?.detail ?? "Loading usage…"))
    let size = NSFont.menuFont(ofSize: 0).pointSize
    let heading = NSFont.monospacedSystemFont(ofSize: size, weight: .semibold)
    let body = NSFont.monospacedSystemFont(ofSize: size, weight: .regular)
    for group in report?.groups ?? [] {
      menu.addItem(.separator())
      menu.addItem(labelItem("[\(group.provider)]", font: heading))
      for line in group.lines { menu.addItem(labelItem(line, font: body)) }
    }
    displayedText = report?.text ?? ""
    menu.addItem(.separator())
    menu.addItem(withTitle: "Copy Report", action: #selector(copyReport), keyEquivalent: "c")
      .target = self
    menu.addItem(.separator())
    let login = menu.addItem(
      withTitle: "Launch at Login", action: #selector(toggleLogin), keyEquivalent: "")
    login.target = self
    if let checkmark {
      login.onStateImage = checkmark
      // Reserve the same gutter when launch at login is off.
      login.offStateImage = NSImage(size: checkmark.size, flipped: false) { _ in true }
    }
    login.state = FileManager.default.fileExists(atPath: loginAgent.path) ? .on : .off
    menu.addItem(withTitle: "Quit", action: #selector(quitApp), keyEquivalent: "q").target = self
  }

  private func labelItem(_ text: String, font: NSFont = .menuFont(ofSize: 0)) -> NSMenuItem {
    let label = NSTextField(labelWithString: text)
    label.font = font
    label.textColor = .labelColor
    let row = NSStackView(views: [label])
    // Match the command rows, including their reserved state column.
    let inset: CGFloat = 16
    let stateWidth = checkmark.map { $0.size.width + 1 } ?? 0
    row.edgeInsets = NSEdgeInsets(top: 4, left: inset + stateWidth, bottom: 4, right: inset)
    row.frame.size = row.fittingSize

    let item = NSMenuItem(title: text, action: nil, keyEquivalent: "")
    item.isEnabled = false
    item.view = row
    return item
  }

  @objc private func copyReport() {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(displayedText, forType: .string)
  }

  // The standard terminate selector can add an automatic menu image.
  @objc private func quitApp() {
    NSApplication.shared.terminate(nil)
  }

  @objc private func toggleLogin() {
    let files = FileManager.default
    if files.fileExists(atPath: loginAgent.path) {
      try? files.removeItem(at: loginAgent)
      return
    }
    let plist: [String: Any] = [
      "Label": loginLabel,
      "ProgramArguments": [executable.path],
      "RunAtLoad": true,
    ]
    try? files.createDirectory(
      at: loginAgent.deletingLastPathComponent(), withIntermediateDirectories: true)
    try? PropertyListSerialization.data(
      fromPropertyList: plist, format: .xml, options: 0
    ).write(to: loginAgent, options: .atomic)
  }

  static func main() {
    if relaunchDetached() { return }
    let cache = home.appending(path: "Library/Caches", directoryHint: .isDirectory)
    try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
    let lock = open(
      cache.appending(path: "tartusage.lock").path, O_RDWR | O_CREAT | O_CLOEXEC, 0o644)
    guard lock >= 0 else {
      perror("tartusage: open lock")
      exit(EXIT_FAILURE)
    }
    defer { close(lock) }
    guard flock(lock, LOCK_EX | LOCK_NB) == 0 else {
      FileHandle.standardError.write(Data("tartusage: already running\n".utf8))
      return
    }

    let app = NSApplication.shared
    let delegate = TartUsage()
    app.delegate = delegate
    app.setActivationPolicy(.accessory)
    withExtendedLifetime(delegate) { app.run() }
  }

  // A new session and redirected stdio let the menu bar app outlive the terminal.
  private static func relaunchDetached() -> Bool {
    guard getppid() != 1, getenv("TARTUSAGE_DETACHED") == nil else { return false }
    var actions: posix_spawn_file_actions_t?
    guard posix_spawn_file_actions_init(&actions) == 0 else { return false }
    defer { posix_spawn_file_actions_destroy(&actions) }
    for fd in STDIN_FILENO...STDERR_FILENO {
      guard posix_spawn_file_actions_addopen(&actions, fd, "/dev/null", O_RDWR, 0) == 0 else {
        return false
      }
    }
    var attributes: posix_spawnattr_t?
    guard posix_spawnattr_init(&attributes) == 0 else { return false }
    defer { posix_spawnattr_destroy(&attributes) }
    guard posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETSID)) == 0,
      setenv("TARTUSAGE_DETACHED", "1", 1) == 0
    else { return false }
    return posix_spawn(
      nil, executable.path, &actions, &attributes, CommandLine.unsafeArgv, environ) == 0
  }
}
