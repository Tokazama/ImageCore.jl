
@testset "StreamingContainer" begin

    struct AVIStream
        dims::NTuple{3,Int}
    end
    Base.ndims(::AVIStream) = 3
    Base.size(A::AVIStream) = A.dims
    AxisIndices.dimnames(::Type{AS}) where {AS<:AVIStream} = (:y, :x, :time)
    Base.axes(A::AVIStream) = (Base.OneTo(A.dims[1]),
                            Base.OneTo(A.dims[2]),
                            Base.OneTo(A.dims[3]))
    function ImageCore.StreamIndexStyle(::Type{AVIStream}, ::Type{typeof(read!)})
        return IndexIncremental()
    end

    P = NamedAxisArray{(:x, :time)}([0 0 0 0; 1 2 3 4; 0 0 0 0])
    f!(dest, a) = (dest[1] = dest[3] = -0.2*a[2]; dest[2] = 0.6*a[2]; dest)
    # Next inference was special-cased for v0.6
    S = @inferred(StreamingContainer{(:time,),Float64}(f!, P))
    @test @inferred(axes(S)) == (Base.OneTo(3), Base.OneTo(4))
    @test @inferred(size(S)) == (3,4)
    @test @inferred(axes(S, 2)) == Base.OneTo(4)
    @test @inferred(size(S, 1)) === 3
    @test @inferred(length(S)) == 12
    @test @inferred(dimnames(S)) == (:x, :time)
    #=
    @test @inferred(axisvalues(S)) === (Base.OneTo(3), Base.OneTo(4))
    @test axisdim(S, Axis{:x}) == axisdim(S, Axis{:x}(1:2)) == axisdim(S, Axis{:x,UnitRange{Int}}) == 1
    @test axisdim(S, Axis{:time}) == 2
    @test_throws ErrorException axisdim(S, Axis{:y})
    @test axisdim(S, Axis{2}) == 2
    @test_throws ErrorException axisdim(S, Axis{3})
    @test @inferred(timeaxis(S)) === Axis{:time}(Base.OneTo(4))
    =#
    @test ntime(S) == 4
    @test @inferred(spatialdims(S)) == (1,)
    @test @inferred(spatial_indices(S)) == (Base.OneTo(3),)
    @test @inferred(spatial_size(S)) == (3,)
    @test @inferred(spatial_order(S)) == (:x,)
    assert_timedim_last(S)
    for i = 1:4
        @test @inferred(S[:, i]) == [-0.2,0.6,-0.2]*i
        @test @inferred(S[2, i]) === 0.6*i
        @test @inferred(S[time = i]) == [-0.2,0.6,-0.2]*i
        @test @inferred(S[time = i, x = 2]) === 0.6*i
        @test @inferred(S[x = 2, time = i]) === 0.6*i
    end
    buf = zeros(3)
    @test @inferred(getindex!(buf, S, :, 2)) == [-0.2,0.6,-0.2]*2
    @test StreamIndexStyle(S) === IndexAny()
    @test StreamIndexStyle(zeros(2,2)) === IndexAny()
    # Non-AbstractArray parent
    #= how should these defaults work?
    @test_throws DimensionMismatch StreamingContainer{UInt8}(read!, AVIStream((1080,1920,10000)), Axis{:foo}())
    V = StreamingContainer{UInt8}(read!, AVIStream((1080,1920,10000)), Axis{:time}())
    @test size(V) == (1080,1920,10000)
    @test dimnames(V) == (:y, :x, :time)
    @test StreamIndexStyle(V) === IndexIncremental()
    =#
    # internal
    # TODO @test ImageAxes.streamingaxisnames(S) == (:time,)
    # TODO @test ImageAxes.filter_streamed((1,2), S) == (2,)
end
