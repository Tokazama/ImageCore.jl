Base.@deprecate_binding ChannelView channelview

export ColorView

struct ColorView{C<:Colorant,N,A<:AbstractArray} <: AbstractArray{C,N}
    parent::A

    function ColorView{C,N,A}(parent::AbstractArray{T}) where {C,N,A,T<:Number}
        n = length(colorview_size(C, parent))
        n == N || throw(DimensionMismatch("for an $N-dimensional ColorView with color type $C, input dimensionality should be $n instead of $(ndims(parent))"))
        checkdim1(C, axes(parent))
        Base.depwarn("ColorView{C}(A) is deprecated, use colorview(C, A)", :ColorView)
        colorview(C, A)
    end
end

function ColorView{C}(A::AbstractArray) where C<:Colorant
    Base.depwarn("ColorView{C}(A) is deprecated, use colorview(C, A)", :ColorView)
    colorview(C, A)
end

ColorView(parent::AbstractArray) = error("must specify the colortype, use colorview(C, A)")

if VERSION < v"1.5"
    Base.@deprecate_binding squeeze1 true
end

# deprecations since "The Great Refactoring"

@deprecate pixelspacing pixel_spacing

@deprecate spacedirections spatial_directions

@deprecate coords_spatial spatialdims

@deprecate indices_spatial spatial_indices

@deprecate size_spatial spatial_size

@deprecate spatialorder spatial_order

@deprecate namedaxes named_axes

@deprecate nimages ntime
