import Darwin
import Foundation
import TOMLKit

private struct Tokens {
  var input: Decimal = 0
  var cached: Decimal = 0
  var output: Decimal = 0

  mutating func add(_ other: Tokens) {
    input += other.input
    cached += other.cached
    output += other.output
  }

  func cost(at rates: Rates) -> Decimal {
    let fresh = max(0, input - cached)
    return (fresh * rates.input + cached * rates.cached + output * rates.output) / 1_000_000
  }
}

private struct LedgerRow: Decodable {
  let model: String
  let stamp: String
  let tokens: Tokens

  private enum CodingKeys: String, CodingKey {
    case model, stamp, input, cached, output
  }

  init(from decoder: any Decoder) throws {
    let row = try decoder.container(keyedBy: CodingKeys.self)
    model = try row.decode(String.self, forKey: .model)
    stamp = (try? row.decode(String.self, forKey: .stamp)) ?? ""
    tokens = try Tokens(
      input: Decimal(row.decode(UInt64.self, forKey: .input)),
      cached: Decimal(row.decode(UInt64.self, forKey: .cached)),
      output: Decimal(row.decode(UInt64.self, forKey: .output))
    )
  }
}

private typealias DailyUsage = [Date?: [String: Tokens]]

actor Ledger {
  private let url: URL
  private var identity: (dev_t, ino_t) = (0, 0)
  private var offset: off_t = 0
  private var anchor = Data()
  private var calendar = Calendar(identifier: .gregorian)
  private var days: DailyUsage = [:]

  init(url: URL) {
    self.url = url
    calendar.timeZone = .current
    calendar.firstWeekday = 1  // Sunday
  }

  func refresh(providers: URL) -> UsageReport {
    readAppendedRows(in: .current)
    return makeReport(days, models: models(in: providers), now: .now, calendar: calendar)
  }

  private func readAppendedRows(in timeZone: TimeZone) {
    if calendar.timeZone != timeZone {
      // Rebucket earlier rows when the Mac moves to another time zone.
      reset()
      calendar.timeZone = timeZone
    }

    guard let file = fopen(url.path, "r") else {
      if errno == ENOENT { reset() }
      return
    }
    defer { fclose(file) }

    let descriptor = fileno(file)
    var info = stat()
    guard fstat(descriptor, &info) == 0 else { return }
    let fileIdentity = (info.st_dev, info.st_ino)
    if fileIdentity != identity || info.st_size < offset || !matchesAnchor(descriptor) {
      reset()
      identity = fileIdentity
    }
    guard offset < info.st_size, fseeko(file, offset, SEEK_SET) == 0 else { return }

    let decoder = JSONDecoder()
    let stamp = Date.ISO8601FormatStyle(timeZone: .gmt)
      .year().month().day().time(includingFractionalSeconds: false)
      .dateSeparator(.dash).dateTimeSeparator(.space).timeSeparator(.colon)
    var line: UnsafeMutablePointer<CChar>?
    var capacity = 0
    defer { free(line) }

    // Commit only newline-terminated rows. A partial write starts here on the next pass.
    while offset < info.st_size {
      let length = getline(&line, &capacity, file)
      guard let line, length > 0, line[length - 1] == 0x0A else { break }
      if let row = try? decoder.decode(LedgerRow.self, from: Data(bytes: line, count: length)) {
        // Invalid stamps still contribute to the all-time model totals.
        let day = (try? stamp.parse(row.stamp)).flatMap {
          stamp.format($0) == row.stamp ? calendar.startOfDay(for: $0) : nil
        }
        days[day, default: [:]][row.model, default: Tokens()].add(row.tokens)
      }
      offset += off_t(length)
      let anchorLength = min(length, 64)
      let suffix = Data(bytes: line.advanced(by: length - anchorLength), count: anchorLength)
      anchor = anchorLength == 64 ? suffix : anchor.suffix(64 - anchorLength) + suffix
    }
  }

  // A short suffix catches typical truncation and regrowth on the same inode.
  private func matchesAnchor(_ descriptor: Int32) -> Bool {
    guard !anchor.isEmpty else { return true }
    var bytes = Data(count: anchor.count)
    let length = bytes.withUnsafeMutableBytes { buffer in
      pread(descriptor, buffer.baseAddress, buffer.count, offset - off_t(buffer.count))
    }
    return length == bytes.count && bytes == anchor
  }

  private func reset() {
    days = [:]
    offset = 0
    anchor = Data()
  }
}

private typealias Rates = (input: Decimal, cached: Decimal, output: Decimal)
private typealias Model = (provider: String, rates: Rates?)
private let posix = Locale(identifier: "en_US_POSIX")

private func rates(in cost: TOMLTable) -> Rates? {
  func rate(_ key: String) -> Decimal? {
    guard let text = cost[key]?.string, let value = Decimal(string: text, locale: posix),
      value >= 0
    else { return nil }
    return value
  }
  guard let input = rate("input"), let cached = rate("cached"), let output = rate("output")
  else { return nil }
  return (input, cached, output)
}

private func models(in url: URL) -> [String: Model] {
  guard let text = try? String(contentsOf: url, encoding: .utf8),
    let providers = try? TOMLTable(string: text)
  else { return [:] }

  var models: [String: Model] = [:]
  for (provider, value) in providers {
    guard let agents = value.table?["agents"]?.array else { continue }
    for agent in agents {
      guard let name = agent["model"]?.string else { continue }
      let first = models[name]
      models[name] = (
        first?.provider ?? provider, first?.rates ?? agent["cost"]?.table.flatMap(rates)
      )
    }
  }
  return models
}

struct UsageReport: Sendable {
  let detail: String
  let groups: [(provider: String, lines: [String])]

  var text: String {
    groups.map { (["[\($0.provider)]"] + $0.lines).joined(separator: "\n") }
      .joined(separator: "\n\n")
  }
}

private let dollars = Decimal.FormatStyle.Currency(code: "USD", locale: Locale(identifier: "en_US"))
  .grouping(.never).rounded(rule: .up)

private func compact(_ count: Decimal) -> String {
  let number = NSDecimalNumber(decimal: count).doubleValue
  for (threshold, scale, suffix) in [(1e12, 1e12, "T"), (1e9, 1e9, "B"), (1e5, 1e6, "M")]
  where number >= threshold {
    return String(format: "%.1f%@", number / scale, suffix)
  }
  return count.description
}

private func makeReport(
  _ days: DailyUsage, models: [String: Model], now: Date, calendar: Calendar
) -> UsageReport {
  guard !days.isEmpty else { return UsageReport(detail: "No usage yet", groups: []) }

  let today = calendar.startOfDay(for: now)
  let week = calendar.dateInterval(of: .weekOfYear, for: now)?.start ?? today
  let month = calendar.dateInterval(of: .month, for: now)?.start ?? today
  let earliest = min(week, month)

  var byModel: [String: Tokens] = [:]
  var dayCost: Decimal = 0
  var weekCost: Decimal = 0
  var monthCost: Decimal = 0
  for (day, bucket) in days {
    for (model, tokens) in bucket {
      byModel[model, default: Tokens()].add(tokens)
      guard let day, day >= earliest, day <= today, let rates = models[model]?.rates else {
        continue
      }
      let cost = tokens.cost(at: rates)
      if day == today { dayCost += cost }
      if day >= week { weekCost += cost }
      if day >= month { monthCost += cost }
    }
  }

  var groups: [String: [String]] = [:]
  for (model, tokens) in byModel.sorted(by: { $0.key < $1.key }) {
    let home = models[model]
    let price = home?.rates.map { "\(tokens.cost(at: $0).formatted(dollars)) " } ?? ""
    let share = tokens.input > 0 ? tokens.cached / tokens.input * 100 : 0
    let percent = String(format: "%.0f", NSDecimalNumber(decimal: share).doubleValue.rounded())
    let counts = "\(compact(tokens.input)) input, \(compact(tokens.output)) output, \(percent)% ⛁"
    groups[home?.provider ?? "unknown", default: []].append("- \(model): \(price)(\(counts))")
  }

  let clock = Date.FormatStyle(locale: Locale(identifier: "en_GB"), timeZone: calendar.timeZone)
    .hour(.twoDigits(amPM: .omitted)).minute(.twoDigits)
  let detail =
    "(\(dayCost.formatted(dollars)) today · \(weekCost.formatted(dollars)) this week · "
    + "\(monthCost.formatted(dollars)) this month) · Updated \(now.formatted(clock))"
  return UsageReport(
    detail: detail,
    groups: groups.sorted(by: { $0.key < $1.key }).map { ($0.key, $0.value) }
  )
}
