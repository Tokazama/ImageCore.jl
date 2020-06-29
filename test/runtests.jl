module ImageCoreTests

using Test
using AxisIndices
using Colors
using MosaicViews
using PaddedViews
pkgs = (AxisIndices, Base, Core, Test, MosaicViews, PaddedViews)
ambs = detect_ambiguities(pkgs...);

using ImageCore
@test isempty(setdiff(detect_ambiguities(ImageCore, pkgs...), ambs))

# FIXME can't use reference tests until it's updated
#using ReferenceTests


using Documenter
DocMeta.setdocmeta!(ImageCore, :DocTestSetup, :(using ImageCore); recursive=true)
doctest(ImageCore, manual = false)

include("colorchannels.jl")
include("views.jl")
include("convert_reinterpret.jl")
include("traits.jl")
include("map.jl")
include("functions.jl")
include("show.jl")

# run these last
isCI = haskey(ENV, "CI") || get(ENV, "JULIA_PKGEVAL", false)
if Base.JLOptions().can_inline == 1 && !isCI
    @info "running benchmarks"
    include("benchmarks.jl")  # these fail if inlining is off
end

end
