# `tartusage`

A native macOS menu bar app that reads `~/.config/tart/usage.jsonl` and shows token
usage and estimated cost per model, grouped by provider.

```text
[openai]
- gpt-6-astra: $3.37 (5.1M input, 0.2M output, 92% ⛁)
- gpt-6.2-sol: $4.99 (16.1M input, 1.3M output, 96% ⛁)

[z.ai]
- glm-5.3: (16.1M input, 1.3M output, 99% ⛁)
```

`⛁` is the share of input tokens served from the prompt cache. Models without a `cost`
in `providers.toml` show their token counts. Counts below 100,000 are shown in full.

The pie icon opens a menu with spend for the current day, week, and month in the Mac's
time zone. Weeks start on Sunday. The `Updated` time uses the same time zone.

```text
($3.37 today · $8.36 this week · $21.00 this month) · Updated 13:45
```

The estimate uses `cost.input` for fresh input, `cost.cached` for cache hits, and
`cost.output` for generated tokens. Rates are in USD per million tokens. The app totals
in Decimal and rounds up to the nearest cent for display.

## Installing

The app targets macOS 13 and later and builds with Swift 6. The first build downloads
the parser for `providers.toml`. [TOMLKit](https://github.com/LebJe/TOMLKit)

```
cd tart-usage
make lint install
$HOME/.local/bin/tartusage # Open the menu-bar app
```

## Resource use

The background reader decodes complete lines with Foundation's `JSONDecoder` and keeps
totals per local day and model. Each refresh resumes at the last complete line, so
partial writes can finish before they add to the count. The scan restarts if the file is
replaced or shortened, if the last 64 bytes already read have changed, or if the Mac's
time zone changes. The app refreshes once per minute.
