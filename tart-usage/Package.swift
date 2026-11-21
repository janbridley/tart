// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "tart-usage",
  platforms: [.macOS(.v13)],
  dependencies: [.package(url: "https://github.com/LebJe/TOMLKit.git", from: "0.6.0")],
  targets: [
    .executableTarget(
      name: "tartusage", dependencies: ["TOMLKit"], path: ".",
      exclude: ["Makefile", "README.md"], sources: ["menubar.swift", "usage.swift"])
  ]
)
