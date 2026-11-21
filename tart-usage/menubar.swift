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
    menu.addItem(withTitle: "tart usage", action: nil, keyEquivalent: "")
    menu.addItem(withTitle: report?.detail ?? "Loading usage…", action: nil, keyEquivalent: "")
    for group in report?.groups ?? [] {
      menu.addItem(.separator())
      menu.addItem(monoItem("[\(group.provider)]", weight: .semibold))
      for line in group.lines { menu.addItem(monoItem(line, weight: .regular)) }
    }
    displayedText = report?.text ?? ""
    menu.addItem(.separator())
    menu.addItem(withTitle: "Copy Report", action: #selector(copyReport), keyEquivalent: "c")
      .target = self
    menu.addItem(.separator())
    let login = menu.addItem(
      withTitle: "Launch at Login", action: #selector(toggleLogin), keyEquivalent: "")
    login.target = self
    if FileManager.default.fileExists(atPath: loginAgent.path) {
      login.image = NSImage(systemSymbolName: "checkmark", accessibilityDescription: "Enabled")
    }
    menu.addItem(
      withTitle: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
  }

  private func monoItem(_ text: String, weight: NSFont.Weight) -> NSMenuItem {
    let font = NSFont.monospacedSystemFont(
      ofSize: NSFont.menuFont(ofSize: 0).pointSize, weight: weight)
    let item = NSMenuItem(title: text, action: nil, keyEquivalent: "")
    item.attributedTitle = NSAttributedString(
      string: text, attributes: [.font: font, .foregroundColor: NSColor.labelColor])
    return item
  }

  @objc private func copyReport() {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(displayedText, forType: .string)
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
    let data = try? PropertyListSerialization.data(
      fromPropertyList: plist, format: .xml, options: 0)
    try? data?.write(to: loginAgent, options: .atomic)
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
