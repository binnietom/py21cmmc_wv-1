subroutine morlet(nx, nt, nf, realdata, imagdata, t, tc, f,rl_out, im_out)
    ! Discrete Morlet Wavelet transform, using Morlet basis from Goupillaud 1984 (Eq. 5, 6 - with b=2pi)
    implicit none

    integer, intent(in)   :: nx, nt, nf
    real(8), intent(in)   :: t(nt), tc(nt), f(nf)
    real(8), intent(out)  :: rl_out(nx,nt,nf), im_out(nx,nt,nf)
    real(8), intent(in)   :: realdata(nx, nt), imagdata(nx,nt)

    integer :: ix, jc,jf, jt, thisn
    real(8) :: exponent, mag, extent, dt

    ! Initialize the output
    rl_out(:,:,:) = 0.d0
    im_out(:,:,:) = 0.d0

    dt = t(2) - t(1)

    do ix=1,nx
        do jf=1,nf
            extent = 1/(f(jf)*sqrt(2.d0))
            thisn = ceiling(5*extent/dt)
            do jc=1,nt
                do jt=max(1, jc-thisn), min(jc+thisn, nt)
                    exponent = f(jf)*(t(jt) - tc(jc))
                    mag = exp(-exponent**2/2) ! Cath has a divide by two here in the CODE only...

                    exponent = exponent*6.2831853071

                    rl_out(ix, jc, jf) = rl_out(ix, jc, jf) + realdata(ix, jt)*mag*cos(exponent)
                    im_out(ix, jc, jf) = im_out(ix, jc, jf) + imagdata(ix, jt)*mag*sin(exponent)

                end do
            end do
        end do
    end do

end subroutine morlet