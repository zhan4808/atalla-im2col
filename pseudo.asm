# phase 0, initial dram -> scpad loads

    # Load input tensor to scpad0
    lui.s    $1, input_addr_hi
    addi.s   $1, $1, input_addr_lo
    lui.s    $2, scpad0_ptr_hi  
    addi.s   $2, $2, scpad0_ptr_lo
    scpad.ld $2, $1, H_in, W_in*C_in

    # Load filter weights to scpad1
    lui.s    $3, filter_addr_hi
    addi.s   $3, $3, filter_addr_lo
    lui.s    $4, scpad1_filter_hi
    addi.s   $4, $4, scpad1_filter_lo
    scpad.ld $4, $3, C_out, C_in*K*K

    # Load/init C tile in scpad1
    lui.s    $5, bias_addr_hi
    addi.s   $5, $5, bias_addr_lo
    lui.s    $6, scpad1_c_hi
    addi.s   $6, $6, scpad1_c_lo
    scpad.ld $6, $5, H_out*W_out, C_out

    barrier.s

# phase 1, main convolution loop
    li.s     $10, 0 # k_h = 0
    li.s     $11, K # K (filter height)

outer_kh_loop:
    li.s     $12, 0 # k_w = 0
    li.s     $13, K # K (filter width)

inner_kw_loop:

    # load input vectors for this (k_h, k_w)
    # base input offset for this tile = k_h * W_in + k_w (simplified)
    mul.s    $20, $10, W_in
    add.s    $20, $20, $12 # $20 = input offset

    # Load 32 input vectors
    li.s     $21, 0 # vec_idx
load_input_loop:
    # compute rcid for this output position (addr gen for implicit im2col) 
    add.s    $22, $20, $21 # rcid
    vreg.ld  $21, $2, $22, ROW # vd=vec_idx, scpad0, rcid
    addi.s   $21, $21, 1
    blt.s    $21, 32, load_input_loop

    # Load filter weights for this (k_h, k_w)
    # Compute filter slice offset
    mul.s    $23, $10, K
    add.s    $23, $23, $12
    mul.s    $23, $23, C_in # $23 = filter slice offset

    li.s     $24, 0  # weight_vec_idx
load_weight_loop:
    add.s    $25, $23, $24 # rcid for this weight row
    addi.s   $26, $24, 32 # vd = 32 + weight_vec_idx
    vreg.ld  $26, $4, $25, ROW
    addi.s   $24, $24, 1
    blt.s    $24, 32, load_weight_loop

    # Load weights into SA
    li.s     $27, 32 # start from VR 32
load_sa_weights_loop:
    lw.vi    $27 # load VR[27] into SA
    addi.s   $27, $27, 1
    blt.s    $27, 64, load_sa_weights_loop

    # Load current C tile (psums)
    li.s     $28, 0
load_psum_loop:
    addi.s   $29, $28, 64  # vd = 64 + psum_idx
    vreg.ld  $29, $6, $28, ROW
    addi.s   $28, $28, 1
    blt.s    $28, 32, load_psum_loop

    # Execute gemm
    li.s     $30, 0
gemm_loop:
    # vs1 = input (VR 0-31)
    # vs2 = psum (VR 64-95)  
    # vd = updated psum (VR 64-95)
    addi.s   $31, $30, 64 # psum reg
    gemm.vv  $31, $30, $31 # vd = vs1 * weights + vs2
    addi.s   $30, $30, 1
    blt.s    $30, 32, gemm_loop

    # Store updated C tile
    li.s     $30, 0
store_psum_loop:
    addi.s   $31, $30, 64
    vreg.st  $31, $6, $30, ROW
    addi.s   $30, $30, 1
    blt.s    $30, 32, store_psum_loop

    addi.s   $12, $12, 1 # k_w++
    blt.s    $12, $13, inner_kw_loop

    addi.s   $10, $10, 1 # k_h++
    blt.s    $10, $11, outer_kh_loop


# Phase 2, store final output to DRAM
    barrier.s
    
    lui.s    $7, output_addr_hi
    addi.s   $7, $7, output_addr_lo
    scpad.st $6, $7, H_out*W_out, C_out

    halt.s