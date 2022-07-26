using MLJFlux, Flux, MLJ, DataFrames, CSV, StatsBase

origindata = CSV.read("data/newyork-city-airbnb-open-data/AB_NYC_2019.csv", DataFrame)

featureSelector = FeatureSelector(
  features = [:id, :name, :host_name, :last_review],
  ignore = true
)

dropMissing(dataframe::DataFrame) = begin
  dropmissing(dataframe, :reviews_per_month)
end

processLongitude(dataframe::DataFrame) = begin
  dataframe[!, :longitude] = map(floor, dataframe[!, :longitude])
  array = unique(dataframe[!, :longitude])
  dict = Dict{Float64, Float64}()
  for (index, value) in Iterators.enumerate(array)
    dict[value] = index
  end

  dataframe[!, :longitude] = map(x -> dict[x], dataframe[!, :longitude])
  return dataframe
end

processNeighbourhoodGroup(dataframe::DataFrame) = begin
  array = unique(dataframe[!, :neighbourhood_group])
  dict = Dict{String, Int}()
  for (index, value) in Iterators.enumerate(array)
    dict[value] = index
  end
  
  dataframe[!, :neighbourhood_group] = map(x -> dict[x], dataframe[!, :neighbourhood_group])

  
  return dataframe
end

processNeighbourhood(dataframe::DataFrame) = begin
  array = unique(dataframe[!, :neighbourhood])

  dict = Dict{String, Int}()
  for (index, value) in Iterators.enumerate(array)
    dict[value] = index
  end
  
  dataframe[!, :neighbourhood] = map(x -> dict[x], dataframe[!, :neighbourhood])

  return dataframe
end

processRoomType(dataframe::DataFrame) = begin
  array = unique(dataframe[!, :room_type])
  dict = Dict{String, Int}()
  for (index, value) in Iterators.enumerate(array)
    dict[value] = index
  end

  dataframe[!, :room_type] = map(x -> dict[x], dataframe[!, :room_type])

  return dataframe
end

coerceCount(dataframe::DataFrame) = begin
  coerce(dataframe, Count => Continuous)
end

# DONE transform data
transformModel = Pipeline(
  featureSelector,
  dropMissing,
  processLongitude,
  processNeighbourhoodGroup,
  processNeighbourhood,
  processRoomType,
  coerceCount
)

transformMachine = machine(transformModel, origindata)
fit!(transformMachine)
transformedData = MLJ.transform(transformMachine, origindata)


using Plots, StatsPlots
plotly()
figuresize = (1200, 900)
# DONE plotting all neighbourhood group
let 
  counts = countmap(origindata[!, :neighbourhood_group])
  bar(collect(keys(counts)), collect(values(counts)),
      title = "Neighbourhood Group",
      size = figuresize) |> display
end

# DONE plotting neighbourhood
let
  counts = countmap(origindata[!, :neighbourhood])
  bar(collect(keys(counts)), collect(values(counts)),
      xrotation = -90,
      xticks = :all,
      size = figuresize,
      title = "Neighbourhood") |> display
end

# DONE plotting room type
let 
  counts = countmap(origindata[!, :room_type])
  bar(collect(keys(counts)), collect(values(counts)), size = figuresize) |> display
end
# DONE plotting relation between neighbourhood_group and availability_365 of room
let
  x = origindata[!, :neighbourhood_group]
  y = origindata[!, :availability_365]
  boxplot(x, y, size = figuresize) |> display
end

# DONE plotting map of neighbourhood_group
let
  array = unique(origindata[!, :neighbourhood_group])
  colors = [:red, :green, :blue, :black, :yellow]
  dict = Dict{String, Symbol}()

  for (index, value) in Iterators.enumerate(array)
    dict[value] = colors[index]
  end

  markercolors = map(x -> dict[x], origindata[!, :neighbourhood_group])
  scatter(origindata[!, :longitude], origindata[!, :latitude],
          markercolor = markercolors,
          size = figuresize) |> display
end
# DOING plotting map of neighbourhood
let
  array = unique(origindata[!, :room_type])
  colors = [:red, :green, :blue]
  dict = Dict{String, Symbol}()
  for (index, value) in Iterators.enumerate(array)
    dict[value] = colors[index]
  end

  markercolors = map(x -> dict[x], origindata[!, :room_type])
  scatter(origindata[!, :longitude], origindata[!, :latitude],
          markercolor = markercolors,
          size = figuresize) |> display
end

# DONE availability of room
let
  mapcolor(number::Number) = begin
    if number >= 0 && number < 150
      return :red
    elseif number >= 150 && number < 300
      return :green
    elseif number >= 300 && number < 450
      return :blue
    else
      return :black
    end
  end

  markercolors = map(mapcolor, origindata[!, :availability_365])
  scatter(origindata[!, :longitude], origindata[!, :latitude],
          markercolor = markercolors,
          size = figuresize) |> display
end

# TODO word cloud
using WordCloud
wc = wordcloud(origindata[!, :neighbourhood]) |> generate!
paint(wc, "/home/steiner/Downloads/neighbourhood.png")
