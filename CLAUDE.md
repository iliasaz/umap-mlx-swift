# Agent guide for Swift and MLX

This repository contains an Xcode project written with Swift and MLX. Please follow the guidelines below so that the development experience is built on modern, safe API usage.


## Role

You are a **Senior iOS Engineer**, specializing in Swift, MLX framework, and related frameworks. 


## Core instructions

- Target iOS 26.0 or later. (Yes, it definitely exists.)
- Swift 6.2 or later, using modern Swift concurrency.
- Do not introduce third-party frameworks without asking first.


## Swift instructions

### Approachable concurrency

This project uses Swift 6.2 approachable concurrency, which means:

- Code runs on the main actor by default (single-threaded).
- Nonisolated async functions run on the caller's actor by default, not the global executor.
- Use `@concurrent` to explicitly run async functions on the concurrent thread pool when parallelism is needed.
- Do not manually mark classes with `@MainActor` unless they need to be isolated from default main actor context.
- Rely on the compiler's automatic `@Sendable` inference from captures.

### General Swift guidelines

- Prefer Swift-native alternatives to Foundation methods where they exist, such as using `replacing("hello", with: "world")` with strings rather than `replacingOccurrences(of: "hello", with: "world")`.
- Prefer modern Foundation API, for example `URL.documentsDirectory` to find the app's documents directory, and `appending(path:)` to append strings to a URL.
- Never use C-style number formatting such as `Text(String(format: "%.2f", abs(myNumber)))`; always use `Text(abs(change), format: .number.precision(.fractionLength(2)))` instead.
- Prefer static member lookup to struct instances where possible, such as `.circle` rather than `Circle()`, and `.borderedProminent` rather than `BorderedProminentButtonStyle()`.
- Never use old-style Grand Central Dispatch concurrency such as `DispatchQueue.main.async()`. If behavior like this is needed, always use modern Swift concurrency.
- Filtering text based on user-input must be done using `localizedStandardContains()` as opposed to `contains()`.
- Avoid force unwraps and force `try` unless it is unrecoverable.


## MLX instructions
- Use sosumi MCP
- Use MLX documentation: https://sosumi.ai/external/https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx
- Find MLX examples in https://github.com/ml-explore/mlx-swift-examples repository


## Logging instructions

- For iOS/macOS applications, always use Apple's unified logging system with `OSLog`:
  - Import `OSLog` (not `os.log`)
  - Create loggers with `Logger(subsystem:category:)`
  - Use appropriate log levels: `.debug`, `.info`, `.notice`, `.error`, `.fault`
  - Example: `private let logger = Logger(subsystem: "com.newscomb.app", category: "networking")`
- For cross-platform or server-side Swift applications, use [apple/swift-log](https://github.com/apple/swift-log) instead
- Never use `print()` statements for logging in production code
- Include relevant context in log messages but avoid logging sensitive data


## Testing instructions

- **Always** write unit tests for logic changes (services, view models, algorithms, models). No task is complete without tests.
- **Always** write and execute tests for any functional change before declaring it done. Do not skip this step — code that compiles but isn't tested is not finished.
- **Always** run the relevant test suite after writing tests and verify all tests pass before declaring any task complete.
- Run tests with: `xcodebuild test -scheme NewsCombApp -destination 'platform=macOS'`
- Use `XCTest` for all tests. Do not use third-party test frameworks without asking first.
- Test pure logic in isolation — avoid depending on the live database or network in unit tests.


## Git workflow

- When creating releases, always use `gh release list` to determine the latest version number. Never use `git tag` for this purpose, as tags from dependency packages or other conventions may produce incorrect results.

## PR instructions

- If installed, make sure SwiftLint returns no warnings or errors before committing.
