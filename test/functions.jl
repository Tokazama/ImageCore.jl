using ImageCore, Colors, FixedPointNumbers
using Base.Test

@testset "functions" begin
    ag = rand(Gray{Float32}, 4, 5)
    ac = rand(RGB{Float32}, 4, 5)
    for (f, args) in ((fft, (ag,)), (fft, (ag, 1:2)), (plan_fft, (ag,)),
                      (rfft, (ag,)), (rfft, (ag, 1:2)), (plan_rfft, (ag,)),
                      (fft, (ac,)), (fft, (ac, 1:2)), (plan_fft, (ac,)),
                      (rfft, (ac,)), (rfft, (ac, 1:2)), (plan_rfft, (ac,)))
        ret = @test_throws ErrorException f(args...)
        @test contains(ret.value.msg, "channelview")
        @test contains(ret.value.msg, eltype(args[1])<:Gray ? "1:2" : "2:3")
    end
    for (a, dims) in ((ag, 1:2), (ac, 2:3))
        @test ifft(fft(channelview(a), dims), dims) ≈ channelview(a)
        ret = @test_throws ErrorException rfft(a)
        @test contains(ret.value.msg, "channelview")
        @test contains(ret.value.msg, "$dims")
        @test irfft(rfft(channelview(a), dims), 4, dims) ≈ channelview(a)
    end
end

nothing