# tartusage: build and install (SwiftPM; first build fetches dependencies).
SWIFT ?= swift
PREFIX ?= $(HOME)/.local
BINDIR := $(PREFIX)/bin
BINARY := .build/release/tartusage
SWIFT_FILES := $(wildcard *.swift)

.PHONY: all clean install lint uninstall

all: $(BINARY)

$(BINARY): $(SWIFT_FILES) Package.resolved
	$(SWIFT) build -c release

install: all
	install -d $(BINDIR)
	install -m 0755 $(BINARY) $(BINDIR)/tartusage

uninstall:
	rm -f $(BINDIR)/tartusage

lint:
	$(SWIFT) format lint --strict $(SWIFT_FILES)
	$(SWIFT) build -Xswiftc -warnings-as-errors
	$(SWIFT) build -c release -Xswiftc -warnings-as-errors

clean:
	rm -rf .build
