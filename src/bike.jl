using MLJFlux, Flux, MLJ, DataFrames, CSV, StatsBase
using Plots, StatsPlots
using Dates
plotly(size = (1600, 1200))

origindata = CSV.read("data/bike-sharing/train.csv", DataFrame)

function transformDataType!(dataframe::DataFrame)
  # 转换日期
  datetimes = map(x -> DateTime(x, "yyyy-mm-dd HH:MM:SS"), dataframe[!, :datetime])

  years = map(year, datetimes)
  months = map(month, datetimes)
  days = map(day, datetimes)
  weekdays = map(dayofweek, datetimes)
  hours = map(hour, datetimes)
  
  dataframe[!, :year] = years
  dataframe[!, :month] = months
  dataframe[!, :weekday] = weekdays
  dataframe[!, :day] = days
  dataframe[!, :hour] = hours


  # 转换季节，工作日，天气
  seasonmap = Dict(1 => "Spring", 2 => "Summer", 3 => "Autumn", 4 => "Winter")
  holidaymap = Dict(1 => "Workingday", 0 => "Weekday")
  weathermap = Dict(1 => "Sunny", 2 => "Cloudy", 3 => "Flurry", 4 => "Heavy Snow")
  dataframe[!, :season] = map(x -> seasonmap[x], dataframe[!, :season])
  dataframe[!, :workingday] = map(x -> holidaymap[x], dataframe[!, :workingday])
  dataframe[!, :weather] = map(x -> weathermap[x], dataframe[!, :weather])
  return dataframe
end

transformDataType!(origindata)

function plotHeatmap(dataframe::DataFrame)
  _schema = schema(dataframe)
  columns = collect(_schema.names)
  scitypes = collect(_schema.scitypes)
  columns = columns[scitypes .!= Textual]

  cormatrix = cor(Matrix(select(dataframe, columns)))
  heatmap(string.(columns), string.(columns), cormatrix) |> display
end

plotHeatmap(origindata)
# MODULE 数据可视化
function plotYearAndCount(dataframe::DataFrame)
  groupDataframe = groupby(dataframe, :year)

  years = Int[]
  counts = Int[]
  for _dataframe in groupDataframe
    year = first(_dataframe[!, :year])
    count = reduce(+, _dataframe[!, :count])

    push!(years, year)
    push!(counts, count)
  end

  bar(years, counts) |> display
end

plotYearAndCount(origindata)

function plotPieOfCount(dataframe::DataFrame)
  totalcount = reduce(+, dataframe[!, :count])
  casualcount = reduce(+, dataframe[!, :casual])
  registeredcount = reduce(+, dataframe[!, :registered])

  xs = ["Casual", "Registered"]
  ys = [casualcount / totalcount, registeredcount / totalcount]

  pie(xs, ys) |> display
end

plotPieOfCount(origindata)

function plotTimeAndCount(dataframe::DataFrame, feature::Symbol)
  features = [:hour, feature]
  groupDataframe = groupby(dataframe, features)
  hours = Int[]
  counts = Float64[]
  features = []
  for _dataframe in groupDataframe
    hour = first(_dataframe[!, :hour])
    count = mean(_dataframe[!, :count])
    _feature = first(_dataframe[!, feature])
    push!(hours, hour)
    push!(counts, count)
    push!(features, _feature)
  end

  partcount = 24
  p = plot()
  for (_hours, _counts, labels) in Iterators.zip(Iterators.partition(hours, partcount),
                                                Iterators.partition(counts, partcount),
                                                Iterators.partition(features, partcount))
    
    indexs = sortperm(_hours)
    xs = _hours[indexs]
    ys = _counts[indexs]
    
    plot!(p, xs, ys, label = first(labels))
  end

  display(p)
end

plotTimeAndCount(origindata, :season)
plotTimeAndCount(origindata, :workingday)
plotTimeAndCount(origindata, :weather)

boxplot(origindata[!, :count]) |> display
boxplot(origindata[!, :hour], origindata[!, :count]) |> display
boxplot(origindata[!, :weekday], origindata[!, :count]) |> display

# DONE plot weather and count
boxplot(origindata[!, :weather], origindata[!, :count]) |> display
# DONE plot season and count
boxplot(origindata[!, :season], origindata[!, :count]) |> display
# TODO plot weather, season and count
# ATTENTION holy shit is this !


# MODULE another notebook
# TODO plot month and count
function plotMonthAndCount(dataframe::DataFrame)
  groupDataframe = groupby(dataframe, [:year, :month])
  months = Int[]
  counts = Int[]
  years = Int[]
  for _dataframe in groupDataframe
    year = first(_dataframe[!, :year])
    month = first(_dataframe[!, :month])
    count = reduce(+, _dataframe[!, :count])
    push!(years, year)
    push!(months, month)
    push!(counts, count)
  end

  partcount = 12
  p = plot()
  for (_months, _counts, year) in Iterators.zip(Iterators.partition(months, partcount),
                                                Iterators.partition(counts, partcount),
                                                sort(unique(years)))
    plot!(p, _months, _counts, label = string(year))
  end
  
  display(p)
end

plotMonthAndCount(origindata)

# TODO plot holiday and count
boxplot(origindata[!, :holiday], origindata[!, :count], xticks = (1:2, ["Non Holi
day", "Holiday"])) |> display

# TODO plot weekday and count
boxplot(origindata[!, :weekday], origindata[!, :count]) |> display
