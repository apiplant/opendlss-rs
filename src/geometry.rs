use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Level {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    pub valid_width: u32,
    pub valid_height: u32,
    pub full_width: u32,
    pub full_height: u32,
    pub levels: [Level; 6],
}

pub fn align_up(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

fn field_alignment(valid: u32) -> u32 {
    let mut reductions = 0;
    let mut size = valid;
    for level in 0..6 {
        let half = align_up(size.div_ceil(2), 4);
        if half < size {
            reductions += 1;
        }
        if level == 0 && !half.is_multiple_of(8) {
            reductions += 1;
        }
        size = half;
    }
    1 << reductions
}

impl Geometry {
    /// Reproduces the native field-padding rule exactly.
    pub fn from_valid(valid_width: u32, valid_height: u32) -> Result<Self> {
        if valid_width == 0 || valid_height == 0 {
            bail!("dimensions must be positive");
        }
        let width_alignment = field_alignment(valid_width);
        let height_alignment = field_alignment(valid_height);
        let mut full_width = 320.max(align_up(valid_width, width_alignment));
        let full_height = 320.max(align_up(valid_height, height_alignment));
        if full_width % (4 * width_alignment) == 0 && full_height % (4 * height_alignment) == 0 {
            full_width += width_alignment;
        }

        let mut width = full_width;
        let mut height = full_height;
        let mut levels = [Level {
            width: 0,
            height: 0,
        }; 6];
        for level in &mut levels {
            width = align_up(width.div_ceil(2), 4);
            height = align_up(height.div_ceil(2), 4);
            *level = Level { width, height };
        }
        if levels[0].width % 8 != 0 || levels[0].height % 8 != 0 {
            bail!(
                "unsupported size {valid_width}x{valid_height}: level 0 is {}x{}; use at least 33 pixels on each axis",
                levels[0].width,
                levels[0].height
            );
        }
        Ok(Self {
            valid_width,
            valid_height,
            full_width,
            full_height,
            levels,
        })
    }

    pub fn vit_tokens(&self) -> u32 {
        self.levels[5].width * self.levels[5].height
    }

    pub fn padded_vit_tokens(&self) -> u32 {
        (self.vit_tokens() + 63) & !63
    }
}

/// The four window views as origin offsets `(shift_x, shift_y)`. Each resolution level runs its own cycle, one
/// step per block at that level in visit order, and a decoder stage continues the count its encoder left.
pub fn window_phase(index: u32) -> (u32, u32) {
    [(0, 0), (4, 4), (4, 0), (0, 4)][(index & 3) as usize]
}

/// Level 6 is the un-pooled field (blocks 0 and 70); 0..5 are the pooled levels.
#[derive(Default)]
pub struct WindowPhases([u32; 7]);

impl WindowPhases {
    pub fn take(&mut self, level: usize) -> u32 {
        let phase = self.0[level];
        self.0[level] += 1;
        phase
    }
}

/// Byte offsets inside one block's weight tensor (FFN -> QKV -> window attention -> projection).
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockLayout {
    pub hidden: u32,
    pub heads: u32,
    pub expert_ffn: bool,
    pub expert_count: u32,
    pub expand: u32,
    pub contract_weights: u32,
    pub input_adapter: u32,
    pub upsample_weight: u32,
    pub transition_scale: u32,
    pub input_scale: u32,
    pub adapter_scale: u32,
    pub post_weights: u32,
    pub ffn_cos_skip: u32,
    pub qkv: u32,
    pub relative: u32,
    pub scale: u32,
    pub projection: u32,
    pub attn_cos_skip: u32,
    pub end_without_padding: u32,
}

fn standard_hidden(channels: u32) -> u32 {
    assert!(
        matches!(channels, 32 | 64 | 128 | 256),
        "no fused layout for {channels} channels"
    );
    128
}

fn ffn_bytes(channels: u32) -> (u32, u32, bool, u32) {
    let hidden = standard_hidden(channels);
    let expert_ffn = channels >= 64;
    let expert_count = if expert_ffn { channels / 32 } else { 0 };
    let expand_bytes = if expert_ffn {
        expert_count * channels * 128
    } else {
        channels * hidden
    };
    let weight_bytes = if expert_ffn {
        expand_bytes + expert_count * 128 * 32 + expert_count * 32 * channels
    } else {
        expand_bytes + hidden * channels
    };
    (expand_bytes, weight_bytes, expert_ffn, expert_count)
}

impl BlockLayout {
    /// A plain block of `channels`, its matrices in order with two 16-byte pads.
    pub fn fused(channels: u32) -> Self {
        let (expand_bytes, weight_bytes, expert_ffn, expert_count) = ffn_bytes(channels);
        let heads = channels / 32;
        let mut l = Self {
            hidden: standard_hidden(channels),
            heads,
            expert_ffn,
            expert_count,
            contract_weights: expand_bytes,
            ..Self::default()
        };
        l.ffn_cos_skip = weight_bytes + 16;
        l.qkv = l.ffn_cos_skip + channels * 2 + 16;
        l.finish_attention(channels);
        l
    }

    /// Block 0: the same block with a 16 -> 32 f16 input adapter in front of it.
    pub fn pre() -> Self {
        Self {
            hidden: 128,
            heads: 1,
            contract_weights: 4096,
            input_adapter: 8208,
            ffn_cos_skip: 9232,
            qkv: 9312,
            relative: 12384,
            scale: 20576,
            projection: 20592,
            attn_cos_skip: 21616,
            end_without_padding: 21680,
            ..Self::default()
        }
    }

    /// The first block of a decoder stage: the 2x upsample weight and the skip scale sit before the QKV.
    pub fn upsample(channels: u32) -> Self {
        let (expand_bytes, weight_bytes, expert_ffn, expert_count) = ffn_bytes(channels);
        let narrow_padding = if channels == 32 { 16 } else { 0 };
        let mut l = Self {
            hidden: standard_hidden(channels),
            heads: channels / 32,
            expert_ffn,
            expert_count,
            contract_weights: expand_bytes,
            upsample_weight: weight_bytes,
            ..Self::default()
        };
        l.ffn_cos_skip = l.upsample_weight + channels * 2 * channels + narrow_padding;
        l.transition_scale = l.ffn_cos_skip + channels * 2 + narrow_padding;
        l.qkv = l.transition_scale + channels * 2;
        l.finish_attention(channels);
        l
    }

    /// Block 70: the post blend's two scales in front, the 32 -> 4 f16 head at the end.
    pub fn post() -> Self {
        Self {
            hidden: 128,
            heads: 1,
            contract_weights: 4096,
            ffn_cos_skip: 8208,
            input_scale: 8272,
            adapter_scale: 8336,
            qkv: 8400,
            relative: 11472,
            scale: 19664,
            projection: 19680,
            attn_cos_skip: 20704,
            post_weights: 20784,
            end_without_padding: 21808,
            ..Self::default()
        }
    }

    fn finish_attention(&mut self, channels: u32) {
        self.relative = self.qkv + channels * channels * 3;
        self.scale = self.relative + self.heads * 8192;
        self.projection = self.scale + align_up(self.heads * 4, 16);
        self.attn_cos_skip = self.projection + channels * channels;
        self.end_without_padding = self.attn_cos_skip + channels * 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_is_window_aligned() {
        let geometry = Geometry::from_valid(1920, 1080).unwrap();
        assert_eq!((geometry.full_width, geometry.full_height), (1920, 1152));
        assert_eq!(
            geometry.levels[0],
            Level {
                width: 960,
                height: 576
            }
        );
        assert_eq!(
            geometry.levels[5],
            Level {
                width: 32,
                height: 20
            }
        );
        assert_eq!(geometry.padded_vit_tokens(), 640);
    }

    #[test]
    fn documented_fields() {
        for ((w, h), (fw, fh)) in [
            ((512, 512), (576, 512)),
            ((768, 768), (832, 768)),
            ((644, 768), (768, 768)),
            ((3840, 2160), (3840, 2176)),
        ] {
            let g = Geometry::from_valid(w, h).unwrap();
            assert_eq!((g.full_width, g.full_height), (fw, fh), "{w}x{h}");
        }
    }

    #[test]
    fn pre_and_post_layouts_match_the_generic_rule() {
        // The hand-written block 0 / 70 layouts share their attention tail with fused(32).
        let fused = BlockLayout::fused(32);
        let pre = BlockLayout::pre();
        assert_eq!(pre.relative - pre.qkv, fused.relative - fused.qkv);
        assert_eq!(
            pre.end_without_padding - pre.projection,
            fused.end_without_padding - fused.projection
        );
    }
}
