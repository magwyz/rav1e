// Copyright (c) 2017-2018, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
pub use self::nasm::get_sad;
#[cfg(any(not(target_arch = "x86_64"), windows, not(feature = "nasm")))]
pub use self::native::get_sad;
use context::{BlockOffset, BLOCK_TO_PLANE_SHIFT, MI_SIZE};
use FrameInvariants;
use FrameState;
use partition::*;
use plane::*;
use rdo::get_lambda_sqrt;

#[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
mod nasm {
  use plane::*;
  use util::*;

  use libc;

  extern {
    fn rav1e_sad_4x4_hbd_ssse3(
      src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
      dst_stride: libc::ptrdiff_t
    ) -> u32;

    fn rav1e_sad_8x8_hbd10_ssse3(
      src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
      dst_stride: libc::ptrdiff_t
    ) -> u32;

    fn rav1e_sad_16x16_hbd_ssse3(
      src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
      dst_stride: libc::ptrdiff_t
    ) -> u32;

    fn rav1e_sad_32x32_hbd10_ssse3(
      src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
      dst_stride: libc::ptrdiff_t
    ) -> u32;

    fn rav1e_sad_64x64_hbd10_ssse3(
      src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
      dst_stride: libc::ptrdiff_t
    ) -> u32;

    fn rav1e_sad_128x128_hbd10_ssse3(
      src: *const u16, src_stride: libc::ptrdiff_t, dst: *const u16,
      dst_stride: libc::ptrdiff_t
    ) -> u32;
  }

  #[target_feature(enable = "ssse3")]
  unsafe fn sad_ssse3(
    plane_org: &PlaneSlice, plane_ref: &PlaneSlice, blk_h: usize,
    blk_w: usize, bit_depth: usize
  ) -> u32 {
    let mut sum = 0 as u32;
    // TODO: stride *2??? What is the correct way to do this?
    let org_stride = plane_org.plane.cfg.stride as libc::ptrdiff_t * 2;
    let ref_stride = plane_ref.plane.cfg.stride as libc::ptrdiff_t * 2;
    assert!(blk_h >= 4 && blk_w >= 4);
    let step_size =
      blk_h.min(blk_w).min(if bit_depth <= 10 { 128 } else { 4 });
    let func = match step_size.ilog() {
      3 => rav1e_sad_4x4_hbd_ssse3,
      4 => rav1e_sad_8x8_hbd10_ssse3,
      5 => rav1e_sad_16x16_hbd_ssse3,
      6 => rav1e_sad_32x32_hbd10_ssse3,
      7 => rav1e_sad_64x64_hbd10_ssse3,
      8 => rav1e_sad_128x128_hbd10_ssse3,
      _ => rav1e_sad_128x128_hbd10_ssse3
    };
    for r in (0..blk_h).step_by(step_size) {
      for c in (0..blk_w).step_by(step_size) {
        let org_slice = plane_org.subslice(c, r);
        let ref_slice = plane_ref.subslice(c, r);
        let org_ptr = org_slice.as_slice().as_ptr();
        let ref_ptr = ref_slice.as_slice().as_ptr();
        sum += func(org_ptr, org_stride, ref_ptr, ref_stride);
      }
    }
    return sum;
  }

  #[inline(always)]
  pub fn get_sad(
    plane_org: &PlaneSlice, plane_ref: &PlaneSlice, blk_h: usize,
    blk_w: usize, bit_depth: usize
  ) -> u32 {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))]
    {
      if is_x86_feature_detected!("ssse3") && blk_h >= 4 && blk_w >= 4 {
        return unsafe {
          sad_ssse3(plane_org, plane_ref, blk_h, blk_w, bit_depth)
        };
      }
    }
    super::native::get_sad(plane_org, plane_ref, blk_h, blk_w, bit_depth)
  }
}

mod native {
  use plane::*;

  #[inline(always)]
  pub fn get_sad(
    plane_org: &PlaneSlice, plane_ref: &PlaneSlice, blk_h: usize,
    blk_w: usize, _bit_depth: usize
  ) -> u32 {
    let mut sum = 0 as u32;

    let org_iter = plane_org.iter_width(blk_w);
    let ref_iter = plane_ref.iter_width(blk_w);

    for (slice_org, slice_ref) in org_iter.take(blk_h).zip(ref_iter) {
      sum += slice_org
        .iter()
        .zip(slice_ref)
        .map(|(&a, &b)| (a as i32 - b as i32).abs() as u32)
        .sum::<u32>();
    }

    sum
  }
}

fn get_mv_range(
  w_in_b: usize, h_in_b: usize, bo: &BlockOffset, blk_w: usize, blk_h: usize
) -> (isize, isize, isize, isize) {
  let border_w = 128 + blk_w as isize * 8;
  let border_h = 128 + blk_h as isize * 8;
  let mvx_min = -(bo.x as isize) * (8 * MI_SIZE) as isize - border_w;
  let mvx_max = (w_in_b - bo.x - blk_w / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_w;
  let mvy_min = -(bo.y as isize) * (8 * MI_SIZE) as isize - border_h;
  let mvy_max = (h_in_b - bo.y - blk_h / MI_SIZE) as isize * (8 * MI_SIZE) as isize + border_h;

  (mvx_min, mvx_max, mvy_min, mvy_max)
}

pub fn get_subset_predictors(
  fi: &FrameInvariants, bo: &BlockOffset, cmv: MotionVector,
  frame_mvs: &Vec<MotionVector>,
  predictors: &mut [Vec<MotionVector>; 3]) {

  // EPZS subset A and B predictors.

  if bo.x > 0 {
    let left = frame_mvs[bo.y * fi.w_in_b + bo.x - 1];
    predictors[1].push(left);
  }
  if bo.y > 0 {
    let top = frame_mvs[(bo.y - 1) * fi.w_in_b + bo.x];
    predictors[1].push(top);

    if bo.x < fi.w_in_b - 1 {
      let top_right = frame_mvs[(bo.y - 1) * fi.w_in_b + bo.x + 1];
      predictors[1].push(top_right);
    }
  }

  if predictors[1].len() > 0 {
    let mut median_mv = MotionVector{row: 0, col: 0};
    for mv in predictors[1].iter() {
      median_mv = median_mv + *mv;
    }
    median_mv = median_mv / (predictors[1].len() as i16);

    predictors[0].push(median_mv.quantize_to_fullpel());
  }

  predictors[1].push(MotionVector{row: 0, col: 0});
  predictors[0].push(cmv.quantize_to_fullpel());

  // EPZS subset C predictors.

  if bo.x > 0 {
    let left = frame_mvs[bo.y * fi.w_in_b + bo.x - 1];
    predictors[2].push(left);
  }
  if bo.y > 0 {
    let top = frame_mvs[(bo.y - 1) * fi.w_in_b + bo.x];
    predictors[2].push(top);
  }
  if bo.x < fi.w_in_b - 1 {
    let right = frame_mvs[bo.y * fi.w_in_b + bo.x + 1];
    predictors[2].push(right);
  }
  if bo.y < fi.h_in_b - 1 {
    let bottom = frame_mvs[(bo.y + 1) * fi.w_in_b + bo.x];
    predictors[2].push(bottom);
  }

  predictors[2].push(frame_mvs[bo.y * fi.w_in_b + bo.x]);
}

pub fn motion_estimation(
  fi: &FrameInvariants, fs: &FrameState, bsize: BlockSize, bo: &BlockOffset,
  ref_frame: usize, cmv: MotionVector, pmv: &[MotionVector; 2],
  frame_mvs: &Vec<MotionVector>
) -> MotionVector {
  match fi.rec_buffer.frames[fi.ref_frames[ref_frame - LAST_FRAME] as usize] {
    Some(ref rec) => {

      let po = PlaneOffset {
        x: (bo.x as isize) << BLOCK_TO_PLANE_SHIFT,
        y: (bo.y as isize) << BLOCK_TO_PLANE_SHIFT
      };
      let blk_w = bsize.width();
      let blk_h = bsize.height();
      let (mut mvx_min, mut mvx_max, mut mvy_min, mut mvy_max) = get_mv_range(fi.w_in_b, fi.h_in_b, bo, blk_w, blk_h);

      // 0.5 is a fudge factor
      let lambda = (get_lambda_sqrt(fi) * 256.0 * 0.5) as u32;

      // Full-pixel motion estimation

      let mut predictors = [Vec::new(), Vec::new(), Vec::new()];

      get_subset_predictors(fi, bo, cmv,
        frame_mvs, &mut predictors);

      let (mut best_mv, mut lowest_cost) = diamond_me_search(
        fi, &po,
        &fs.input.planes[0], &rec.frame.planes[0],
        &predictors, fi.sequence.bit_depth,
        pmv, lambda,
        mvx_min, mvx_max, mvy_min, mvy_max,
        blk_w, blk_h);

      // Sub-pixel motion estimation

      let mode = PredictionMode::NEWMV;
      let mut tmp_plane = Plane::new(blk_w, blk_h, 0, 0, 0, 0);

      let mut steps = vec![8, 4, 2];
      if fi.allow_high_precision_mv {
        steps.push(1);
      }

      for step in steps {
        let center_mv_h = best_mv;
        for i in 0..3 {
          for j in 0..3 {
            // Skip the center point that was already tested
            if i == 1 && j == 1 {
              continue;
            }

            let cand_mv = MotionVector {
              row: center_mv_h.row + step * (i as i16 - 1),
              col: center_mv_h.col + step * (j as i16 - 1)
            };

            if (cand_mv.col as isize) < mvx_min || (cand_mv.col as isize) > mvx_max {
              continue;
            }
            if (cand_mv.row as isize) < mvy_min || (cand_mv.row as isize) > mvy_max {
              continue;
            }

            {
              let tmp_slice =
                &mut tmp_plane.mut_slice(&PlaneOffset { x: 0, y: 0 });

              mode.predict_inter(
                fi,
                0,
                &po,
                tmp_slice,
                blk_w,
                blk_h,
                [ref_frame, NONE_FRAME],
                [cand_mv, MotionVector { row: 0, col: 0 }]                
              );
            }

            let plane_org = fs.input.planes[0].slice(&po);
            let plane_ref = tmp_plane.slice(&PlaneOffset { x: 0, y: 0 });

            let sad = get_sad(&plane_org, &plane_ref, blk_h, blk_w, fi.sequence.bit_depth);

            let rate1 = get_mv_rate(cand_mv, pmv[0], fi.allow_high_precision_mv);
            let rate2 = get_mv_rate(cand_mv, pmv[1], fi.allow_high_precision_mv);
            let rate = rate1.min(rate2 + 1);
            let cost = 256 * sad + rate * lambda;

            if cost < lowest_cost {
              lowest_cost = cost;
              best_mv = cand_mv;
            }
          }
        }
      }

      best_mv
    }

    None => MotionVector { row: 0, col: 0 }
  }
}

fn get_best_predictor(fi: &FrameInvariants,
  po: &PlaneOffset, p_org: &Plane, p_ref: &Plane,
  predictors: &[MotionVector],
  bit_depth: usize, pmv: &[MotionVector; 2], lambda: u32,
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  blk_w: usize, blk_h: usize) -> (MotionVector, u32) {
  let mut center_mv = MotionVector{row: 0, col: 0};
  let mut center_mv_cost = std::u32::MAX;

  for init_mv in predictors.iter() {
    let cost = get_mv_rd_cost(
      fi, po, p_org, p_ref, bit_depth,
      pmv, lambda, mvx_min, mvx_max, mvy_min, mvy_max,
      blk_w, blk_h, *init_mv);

    if cost < center_mv_cost {
      center_mv = *init_mv;
      center_mv_cost = cost;
    }
  }

  return (center_mv, center_mv_cost);
}

fn diamond_me_search(
  fi: &FrameInvariants,
  po: &PlaneOffset, p_org: &Plane, p_ref: &Plane,
  predictors: &[Vec<MotionVector>; 3],
  bit_depth: usize, pmv: &[MotionVector; 2], lambda: u32,
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  blk_w: usize, blk_h: usize) -> (MotionVector, u32)
{
  let diamond_pattern = [(1i16, 0i16), (0, 1), (-1, 0), (0, -1)];
  let mut diamond_radius: i16 = 16;

  let (mut center_mv, mut center_mv_cost) = get_best_predictor(
    fi, po, p_org, p_ref, &predictors[0],
    bit_depth, pmv, lambda, mvx_min, mvx_max, mvy_min, mvy_max,
    blk_w, blk_h);

  if center_mv_cost / ((blk_w * blk_h) as u32) < 256 {
    return (center_mv, center_mv_cost / ((blk_w * blk_h) as u32))
  }

  if predictors[1].len() > 0 {
    let (test_center_mv, test_center_mv_cost) = get_best_predictor(
      fi, po, p_org, p_ref, &predictors[1],
      bit_depth, pmv, lambda, mvx_min, mvx_max, mvy_min, mvy_max,
      blk_w, blk_h);

    if test_center_mv_cost < center_mv_cost {
      center_mv_cost = test_center_mv_cost;
      center_mv = test_center_mv;
    }
  }

  if predictors[2].len() > 0 {
    let (test_center_mv, test_center_mv_cost) = get_best_predictor(
      fi, po, p_org, p_ref, &predictors[2],
      bit_depth, pmv, lambda, mvx_min, mvx_max, mvy_min, mvy_max,
      blk_w, blk_h);

    if test_center_mv_cost < center_mv_cost {
      center_mv_cost = test_center_mv_cost;
      center_mv = test_center_mv;
    }
  }

  loop {
    let mut best_diamond_rd_cost = std::u32::MAX;
    let mut best_diamond_mv = MotionVector { row: 0, col: 0 };

    for p in diamond_pattern.iter() {

        let cand_mv = MotionVector {
          row: center_mv.row + diamond_radius * p.0,
          col: center_mv.col + diamond_radius * p.1
        };

        let rd_cost = get_mv_rd_cost(
          fi, &po, p_org, p_ref, bit_depth,
          pmv, lambda, mvx_min, mvx_max, mvy_min, mvy_max,
          blk_w, blk_h, cand_mv);

        if rd_cost < best_diamond_rd_cost {
          best_diamond_rd_cost = rd_cost;
          best_diamond_mv = cand_mv;
        }
    }

    if center_mv_cost <= best_diamond_rd_cost {
      if diamond_radius == 8 {
        break;
      } else {
        diamond_radius /= 2;
      }
    }
    else {
      center_mv = best_diamond_mv;
      center_mv_cost = best_diamond_rd_cost;
    }
  }

  assert!(center_mv_cost < std::u32::MAX);

  (center_mv, center_mv_cost)
}

fn get_mv_rd_cost(
  fi: &FrameInvariants,
  po: &PlaneOffset, p_org: &Plane, p_ref: &Plane, bit_depth: usize,
  pmv: &[MotionVector; 2], lambda: u32,
  mvx_min: isize, mvx_max: isize, mvy_min: isize, mvy_max: isize,
  blk_w: usize, blk_h: usize,
  cand_mv: MotionVector) -> u32
{
  if (cand_mv.col as isize) < mvx_min || (cand_mv.col as isize) > mvx_max {
    return std::u32::MAX;
  }
  if (cand_mv.row as isize) < mvy_min || (cand_mv.row as isize) > mvy_max {
    return std::u32::MAX;
  }

  let plane_org = p_org.slice(po);
  let plane_ref = p_ref.slice(&PlaneOffset {
    x: po.x + (cand_mv.col / 8) as isize,
    y: po.y + (cand_mv.row / 8) as isize
  });

  let sad = get_sad(&plane_org, &plane_ref, blk_h, blk_w, bit_depth);

  let rate1 = get_mv_rate(cand_mv, pmv[0], fi.allow_high_precision_mv);
  let rate2 = get_mv_rate(cand_mv, pmv[1], fi.allow_high_precision_mv);
  let rate = rate1.min(rate2 + 1);

  return 256 * sad + rate * lambda;
}

// Adjust block offset such that entire block lies within frame boundaries
fn adjust_bo(bo: &BlockOffset, fi: &FrameInvariants, blk_w: usize, blk_h: usize) -> BlockOffset {
  BlockOffset {
    x: (bo.x as isize).min(fi.w_in_b as isize - blk_w as isize / 4).max(0) as usize,
    y: (bo.y as isize).min(fi.h_in_b as isize - blk_h as isize / 4).max(0) as usize
  }
}

fn get_mv_rate(a: MotionVector, b: MotionVector, allow_high_precision_mv: bool) -> u32 {
  fn diff_to_rate(diff: i16, allow_high_precision_mv: bool) -> u32 {
    let d = if allow_high_precision_mv { diff } else { diff >> 1 };
    if d == 0 {
      0
    } else {
      2 * (16 - d.abs().leading_zeros())
    }
  }

  diff_to_rate(a.row - b.row, allow_high_precision_mv) + diff_to_rate(a.col - b.col, allow_high_precision_mv)
}

pub fn estimate_motion_ss4(
  fi: &FrameInvariants, fs: &FrameState, bsize: BlockSize, ref_idx: usize,
  bo: &BlockOffset
) -> Option<MotionVector> {
  if let Some(ref rec) = fi.rec_buffer.frames[ref_idx] {
    let blk_w = bsize.width();
    let blk_h = bsize.height();
    let mut bo_adj = adjust_bo(bo, fi, blk_w, blk_h);
    bo_adj.x = bo_adj.x >> 2;
    bo_adj.y = bo_adj.y >> 2;
    let po = PlaneOffset {
      x: (bo_adj.x as isize) << BLOCK_TO_PLANE_SHIFT,
      y: (bo_adj.y as isize) << BLOCK_TO_PLANE_SHIFT
    };
    let (mvx_min, mvx_max, mvy_min, mvy_max) = get_mv_range(
      fi.w_in_b >> 2, fi.h_in_b >> 2,
      &bo_adj,
      blk_w >> 2, blk_h >> 2);

    // Divide by 16 to account for subsampling, 0.125 is a fudge factor
    let lambda = (get_lambda_sqrt(fi) * 256.0 / 16.0 * 0.125) as u32;

    let mut predictors = [
      vec!(MotionVector{row: 0, col: 0}),
      Vec::new(),
      Vec::new()];

    let (mut best_mv, mut _lowest_cost) = diamond_me_search(
      fi, &po,
      &fs.input_qres, &rec.input_qres,
      &predictors, fi.sequence.bit_depth,
      &[MotionVector { row: 0, col: 0 }; 2], lambda,
      mvx_min, mvx_max, mvy_min, mvy_max,
      blk_w >> 2, blk_h >> 2);

    Some(MotionVector { row: best_mv.row * 4, col: best_mv.col * 4 })
  } else {
    None
  }
}

pub fn estimate_motion_ss2(
  fi: &FrameInvariants, fs: &FrameState, bsize: BlockSize, ref_idx: usize,
  bo: &BlockOffset, pmvs: &[Option<MotionVector>; 3]
) -> Option<MotionVector> {
  if let Some(ref rec) = fi.rec_buffer.frames[ref_idx] {
    let blk_w = bsize.width();
    let blk_h = bsize.height();
    let mut bo_adj = adjust_bo(bo, fi, blk_w, blk_h);
    bo_adj.x = bo_adj.x >> 1;
    bo_adj.y = bo_adj.y >> 1;
    let po = PlaneOffset {
      x: (bo_adj.x as isize) << BLOCK_TO_PLANE_SHIFT,
      y: (bo_adj.y as isize) << BLOCK_TO_PLANE_SHIFT
    };
    let (mvx_min, mvx_max, mvy_min, mvy_max) = get_mv_range(
      fi.w_in_b >> 1, fi.h_in_b >> 1,
      &bo_adj,
      blk_w >> 1, blk_h >> 1);
    let mut best_mv = MotionVector { row: 0, col: 0 };

    // Divide by 4 to account for subsampling, 0.125 is a fudge factor
    let lambda = (get_lambda_sqrt(fi) * 256.0 / 4.0 * 0.125) as u32;

    for omv in pmvs.iter() {
      if let Some(pmv) = omv {
        let mut predictors = [
          vec!(pmv.clone()),
          Vec::new(),
          Vec::new()];

        let ret = diamond_me_search(
          fi, &po,
          &fs.input_hres, &rec.input_hres,
          &predictors, fi.sequence.bit_depth,
          &[MotionVector { row: 0, col: 0 }; 2], lambda,
          mvx_min, mvx_max, mvy_min, mvy_max,
          blk_w >> 1, blk_h >> 1);

        best_mv = ret.0;
      }
    }

    Some(MotionVector { row: best_mv.row * 2, col: best_mv.col * 2 })
  } else {
    None
  }
}

#[cfg(test)]
pub mod test {
  use super::*;
  use partition::BlockSize;
  use partition::BlockSize::*;

  // Generate plane data for get_sad_same()
  fn setup_sad() -> (Plane, Plane) {
    let mut input_plane = Plane::new(640, 480, 0, 0, 128 + 8, 128 + 8);
    let mut rec_plane = input_plane.clone();

    for (i, row) in input_plane.data.chunks_mut(input_plane.cfg.stride).enumerate() {
      for (j, mut pixel) in row.into_iter().enumerate() {
        let val = ((j + i) as i32 & 255i32) as u16;
        assert!(val >= u8::min_value().into() &&
            val <= u8::max_value().into());
        *pixel = val;
      }
    }

    for (i, row) in rec_plane.data.chunks_mut(rec_plane.cfg.stride).enumerate() {
      for (j, mut pixel) in row.into_iter().enumerate() {
        let val = (j as i32 - i as i32 & 255i32) as u16;
        assert!(val >= u8::min_value().into() &&
            val <= u8::max_value().into());
        *pixel = val;
      }
    }

    (input_plane, rec_plane)
  }

  // Regression and validation test for SAD computation
  #[test]
  fn get_sad_same() {
    let blocks: Vec<(BlockSize, u32)> = vec![
      (BLOCK_4X4, 1912),
      (BLOCK_4X8, 3496),
      (BLOCK_8X4, 4296),
      (BLOCK_8X8, 7824),
      (BLOCK_8X16, 14416),
      (BLOCK_16X8, 16592),
      (BLOCK_16X16, 31136),
      (BLOCK_16X32, 59552),
      (BLOCK_32X16, 60064),
      (BLOCK_32X32, 120128),
      (BLOCK_32X64, 250176),
      (BLOCK_64X32, 186688),
      (BLOCK_64X64, 438912),
      (BLOCK_64X128, 1016768),
      (BLOCK_128X64, 654272),
      (BLOCK_128X128, 1689792),
      (BLOCK_4X16, 6664),
      (BLOCK_16X4, 8680),
      (BLOCK_8X32, 27600),
      (BLOCK_32X8, 31056),
      (BLOCK_16X64, 116384),
      (BLOCK_64X16, 93344),
    ];

    let bit_depth: usize = 8;
    let (input_plane, rec_plane) = setup_sad();

    for block in blocks {
      let bsw = block.0.width();
      let bsh = block.0.height();
      let po = PlaneOffset { x: 40, y: 40 };

      let mut input_slice = input_plane.slice(&po);
      let mut rec_slice = rec_plane.slice(&po);

      assert_eq!(
        block.1,
        get_sad(&mut input_slice, &mut rec_slice, bsw, bsh, bit_depth)
      );
    }
  }
}
