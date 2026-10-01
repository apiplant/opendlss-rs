# Packaging

One package, `opendlss-rs`, carrying `opendlss-nr`.
The `packages`, `homebrew`, `apt` and `pacman` jobs in
[`.github/workflows/release.yml`](../.github/workflows/release.yml) substitute
the `@VERSION@`, `@SHA_*@` and `@ARCH@` placeholders with the tag's version and
the checksums of the archives that release just built, then publish the
result. None of the definitions compiles the project from source — they all
install the binaries the build jobs already produced, which is what keeps the
packages and the release byte-identical.

| File | Publishes to |
| --- | --- |
| `homebrew/opendlss-rs.rb` | `apiplant/homebrew-tap`, as `Formula/opendlss-rs.rb` |
| `pacman/PKGBUILD` | the release itself, as a `.pkg.tar.zst` asset, and `apiplant/pacman` |
| `debian/control` | the release itself, as `.deb` assets, and `apiplant/apt` |
| `apt/apt-ftparchive.conf` | `apiplant/apt`, served at `apt.apiplant.com` |

This repository reuses the shared `apiplant` repositories that already serve
`gliner-rs`, `laya-rs`, `nvidia-smi-live`, `portward` and `megadl-rs` —
`apiplant/homebrew-tap`, `apiplant/pacman` (served at
`apiplant.github.io/pacman`) and `apiplant/apt` (served at `apt.apiplant.com`)
— rather than standing up new ones. The same repository secrets those releases
use (`HOMEBREW_TAP_TOKEN`, `PACMAN_REPO_TOKEN` +
`PACMAN_GPG_PRIVATE_KEY`/`PACMAN_GPG_PASSPHRASE`, `APT_REPO_TOKEN` +
`APT_GPG_PRIVATE_KEY`/`APT_GPG_PASSPHRASE`) work here unchanged. They are copied
here by the `Sync release secrets` workflow in `apiplant/gliner-rs`; run it
once (with `dry_run` off) if this repository doesn't have them yet. A publish
job whose credential is absent is skipped rather than failing the release, so
a fork — or this repository before the secrets are copied — still gets a clean
release.

## Platform matrix

| Platform | How it ships |
| --- | --- |
| macOS (Apple Silicon, `aarch64-apple-darwin`) | archive + Homebrew formula |
| Linux x86_64 (`x86_64-unknown-linux-gnu`) | archive, `.deb` + apt repo, Arch package + pacman repo, Homebrew formula |
| Linux aarch64 (`aarch64-unknown-linux-gnu`) | archive, `.deb` + apt repo, Homebrew formula |

The Arch package is x86_64 only: it would need an arm runner or emulation to
build, and the Arch repository has no aarch64 audience. The `.deb`, the apt
repo and the plain archive still cover Linux arm64.

There is no separate CUDA variant for this project: the CUDA backend is built in (its kernels ship as embedded PTX and the driver is loaded at run time), so the one binary uses CUDA when an NVIDIA driver is present and falls back to wgpu otherwise (`--backend auto`).

The order is: build every archive, build the distro packages from those
archives, publish the release, then publish to Homebrew, apt and pacman — the
formula, apt pool and PKGBUILD all reference (or are built from) the archives
that release holds, and the GitHub release should exist first so the URLs they
link stay valid.

## Cutting a release

Bump `version` in `Cargo.toml`, commit, then:

```bash
git tag v$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2) && git push origin --tags
```

`workflow_dispatch` runs the build and packaging jobs without publishing
anything (a dry run).

## Using the packages

macOS (Apple Silicon) and Linux, via Homebrew:

```bash
brew tap apiplant/tap
brew install apiplant/tap/opendlss-rs
```

Arch Linux, via the signed pacman repository at `apiplant.github.io/pacman`
(one-time setup, then `pacman -Sy`/`-Syu` picks up new releases):

```bash
curl -sSfL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg
keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ { print $5; exit }') && sudo pacman-key --add /tmp/apiplant.gpg && sudo pacman-key --finger "$keyid" && sudo pacman-key --lsign-key "$keyid"
printf '\n[apiplant]\nSigLevel = Required DatabaseOptional\nServer = https://apiplant.github.io/pacman/$arch\n' | sudo tee -a /etc/pacman.conf > /dev/null
sudo pacman -Sy opendlss-rs
```

Debian/Ubuntu, via the signed apt repository at `apt.apiplant.com` (one-time
setup, then `apt upgrade` picks up new releases):

```bash
curl -sSfL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main" | sudo tee /etc/apt/sources.list.d/apiplant.list > /dev/null
sudo apt update && sudo apt install opendlss-rs
```

Or install a single `.deb`/`.pkg.tar.zst` release asset directly without adding
a repository:

```bash
sudo dpkg -i opendlss-rs_*_amd64.deb
sudo pacman -U opendlss-rs-*-x86_64.pkg.tar.zst
```

Or download and unpack the archive for your platform from the release page — no
installation required.
