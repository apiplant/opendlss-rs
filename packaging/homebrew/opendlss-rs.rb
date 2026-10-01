# Generated from packaging/homebrew/opendlss-rs.rb in apiplant/opendlss-rs by the
# release workflow, which fills in the version and checksums and commits the
# result to apiplant/homebrew-tap as Formula/opendlss-rs.rb. Changes belong in
# the source repository: the next release overwrites this file.
class OpendlssRs < Formula
  desc "Rust host and model tools for the OpenDLSS-NR neural renderer"
  homepage "https://github.com/apiplant/opendlss-rs"
  version "@VERSION@"
  license "MIT"

  # No bottles: the release archives *are* the binaries, so the formula only
  # unpacks what the tagged workflow already built for each platform.
  on_macos do
    on_arm do
      url "https://github.com/apiplant/opendlss-rs/releases/download/v@VERSION@/opendlss-rs-v@VERSION@-aarch64-apple-darwin.tar.gz"
      sha256 "@SHA_MACOS_ARM64@"
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/apiplant/opendlss-rs/releases/download/v@VERSION@/opendlss-rs-v@VERSION@-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_X86_64@"
    end
    on_arm do
      url "https://github.com/apiplant/opendlss-rs/releases/download/v@VERSION@/opendlss-rs-v@VERSION@-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_ARM64@"
    end
  end

  def install
    bin.install "opendlss-nr"
    doc.install "README.md"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/opendlss-nr --version")
  end
end
