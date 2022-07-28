using MLJFlux, Flux, MLJ, DataFrames, CSV, StatsBase, Dates
import JSON

using Plots
plotly()
figuresize = (1600, 1200)
creditsdata = CSV.read("data/movies/tmdb_5000_credits.csv", DataFrame)
moviesdata = CSV.read("data/movies/tmdb_5000_movies.csv", DataFrame)

first(creditsdata, 5)
first(moviesdata, 5)

# DONE 合并数据集
select!(creditsdata, Not(:title))
fulldata = hcat(creditsdata, moviesdata)


# MODULE 数据清洗
# DONE 选择子集
columns = [:id, :title, :vote_average, :production_companies, :genres,
           :release_date, :keywords, :runtime, :budget, :revenue, :vote_count, :popularity]
fulldata = select(fulldata, columns)
# DONE 缺失值处理
fillReleaseDate(dataframe::DataFrame) = begin
  mapdate(date::Union{Missing, Date}) = begin
    if ismissing(date)
      return Date("2014-06-01")
    else
      return date
    end
  end

  dataframe[!, :release_date] = map(mapdate, dataframe[!, :release_date])
  return dataframe
end

fillRuntime(dataframe::DataFrame) = begin
  meanvalue = mean(skipmissing(dataframe[!, :runtime]))
  mapruntime(runtime::Union{Missing, Float64}) = begin
    if ismissing(runtime)
      return meanvalue
    else
      return runtime
    end
  end

  dataframe[!, :runtime] = map(mapruntime, dataframe[!, :runtime])
  return dataframe
end

generateReleaseYear(dataframe::DataFrame) = begin
  dataframe[!, :release_year] = map(year, dataframe[!, :release_date])
  return dataframe
end

generateName(dataframe::DataFrame) = begin
  mapfn(array::Vector{Any}) = join(map(x -> x["name"], array), ",")

  let
    jsons = map(JSON.parse, dataframe[!, :genres])
    dataframe[!, :genres] = map(mapfn, jsons)
  end

  let 
    jsons = map(JSON.parse, dataframe[!, :production_companies])
    dataframe[!, :production_companies] = map(mapfn, jsons)
  end

  let
    jsons = map(JSON.parse, dataframe[!, :keywords])
    dataframe[!, :keywords] = map(mapfn, jsons)
  end
  return dataframe
end

function fetchGenreList(dataframe::DataFrame)
  mapfn(array::Vector{Any}) = map(x -> x["name"], array)
  genrelist = Set{String}()
  jsons = map(JSON.parse, dataframe[!, :genres])
  for json in jsons
    names = mapfn(json)
    for name in names
      push!(genrelist, name)
    end
  end
  return genrelist
end

const genrelist = fetchGenreList(fulldata)

# DONE 将电影类型添加到列，需进行one-hot编码
function generateGenreType(dataframe::DataFrame)
  len = first(size(dataframe))

  for column in genrelist
    dataframe[!, column] = zeros(len)
    for row in eachrow(dataframe)
      if contains(row.genres, column)
        row[column] = 1
      end
    end
  end

  return dataframe
end

# DONE 用年份索引
function sortByReleaseYear(dataframe::DataFrame)
  sort(dataframe, [:release_year])
end

# ATTENTION this is ok
# featureSelector(dataframe::DataFrame) = begin
#   select(dataframe, Not(:release_date))
# end

featureSelector = FeatureSelector(
  features = [:release_date],
  ignore = true
)

transformModel = Pipeline(
  fillReleaseDate,
  fillRuntime,
  generateReleaseYear,
  featureSelector,
  generateGenreType,
  sortByReleaseYear
  # generateName
)

transformMachine = machine(transformModel, fulldata)

fit!(transformMachine)
transformedData = MLJ.transform(transformMachine, fulldata)

# DONE 对每个类型的电影按年份求和
function groupByReleaseYear(dataframe::DataFrame)
  dataframes = groupby(dataframe, :release_year)
  years = Int[]
  counts = Int[]
  for _dataframe in dataframes
    year = first(_dataframe.release_year)
    count = first(size(_dataframe))

    push!(years, year)
    push!(counts, count)
  end

  bar(years, counts, xticks = :all, size = figuresize) |> display
end

groupByReleaseYear(transformedData)

# DONE 汇总各电影类型的总量
function groupByEachGenre(dataframe::DataFrame)
  record = Dict{String, Int}()
  for genre in genrelist
    record[genre] = 0
  end

  for row in eachrow(dataframe)
    for genre in genrelist
      record[genre] += row[genre]
    end
  end

  _xs = collect(keys(record))
  _ys = collect(values(record))
  indexs = sortperm(_ys)
  xs = _xs[indexs]
  ys = _ys[indexs]
  bar(xs, ys, xticks = :all, size = figuresize) |> display
end

groupByEachGenre(transformedData)

# DONE 电影类型随时间的变化
function plotGenreAndTime(dataframe::DataFrame)
  columns = ["Drama","Comedy","Thriller","Action","Romance","Adventure",
             "Crime","Science Fiction","Horror","Family", "release_year"]
  _dataframes = groupby(select(dataframe, columns), :release_year)
  # p = plot()
  
  # record: Dict{year, Dict{Name, Count}}
  record = Dict{Int, Dict{String, Int}}()
  for _dataframe in _dataframes
    # years
    # counts
    year = first(_dataframe.release_year)
    record[year] = Dict{String, Int}()
    for column in columns[columns .!= "release_year"]
      record[year][column] = reduce(+, _dataframe[!, column])
    end
  end
  
  _years = collect(keys(record))
  _countmaps = collect(values(record))
  indexs = sortperm(_years)

  years = _years[indexs]
  countmaps = _countmaps[indexs]

  p = plot(size = figuresize, xticks = :all)
  for column in columns[columns .!= "release_year"]
    counts = map(x -> x[column], countmaps)
    plot!(p, years, counts, label = column, xticks = :all)
  end

  plot(p) |> display
end

plotGenreAndTime(transformedData)

# DONE 影响电影收入的客观因素有哪些

# MODULE Universal Pictures 和 Paramount Pictures 之间的对比

# DONE 电影发行量对比
function plotCompareTotal(dataframe::DataFrame)
  dataframe[!, "Universal Pictures"] = map(s -> contains(s, "Universal Pictures") ? 1 : 0, dataframe[!, :production_companies])
  dataframe[!, "Paramount Pictures"] = map(s -> contains(s, "Paramount Pictures") ? 1 : 0, dataframe[!, :production_companies])

  universalTotal = reduce(+, dataframe[!, "Universal Pictures"])
  paramountTotal = reduce(+, dataframe[!, "Paramount Pictures"])
  total = universalTotal + paramountTotal

  xs = ["Universal Pictures", "Paramount Pictures"]
  ys = [universalTotal / total, paramountTotal / total]
  pie(xs, ys, aspect_ratio = 1.0) |> display

  companyDifference = groupby(select(dataframe, vcat(xs, "release_year")), :release_year)
  # record: Dict{Year, Dict{Company, Int}}
  record = Dict{Int, Dict{String, Int}}()
  for _dataframe in companyDifference
    year = first(_dataframe.release_year)
    record[year] = Dict{String, Int}()
    for column in xs
      count = reduce(+, _dataframe[!, column])
      record[year][column] = count
    end
  end

  _years = collect(keys(record))
  _countmaps = collect(values(record))
  indexs = sortperm(_years)

  years = _years[indexs]
  countmaps = _countmaps[indexs]

  p = plot(size = figuresize)
  for column in xs
    counts = map(x -> x[column], countmaps)
    plot!(p, years, counts, label = column, xticks = :all)
  end

  plot(p) |> display
end

plotCompareTotal(transformedData)

# DONE 利润对比
function plotCompareProfit(dataframe::DataFrame)
  dataframe[!, :profit] = dataframe[!, :revenue] .- dataframe[!, :budget]
  dataframe[!, "Universal Profit"] = dataframe[!, "Universal Pictures"] .* dataframe[!, :profit]
  dataframe[!, "Paramount Profit"] = dataframe[!, "Paramount Pictures"] .* dataframe[!, :profit]

  universalProfit = reduce(+, dataframe[!, "Universal Profit"])
  paramountProfit = reduce(+, dataframe[!, "Paramount Profit"])
  totalProfit = universalProfit + paramountProfit

  xs = ["Universal Profit", "Paramount Profit"]
  ys = [universalProfit / totalProfit, paramountProfit / totalProfit]
  pie(xs, ys) |> display

  companyDifference = groupby(select(dataframe, vcat(xs, "release_year")), :release_year)
  # record: Dict{Year, Dict{Company, Number}}
  record = Dict{Int, Dict{String, Number}}()
  for _dataframe in companyDifference
    year = first(_dataframe.release_year)
    record[year] = Dict{String, Number}()
    for column in xs
      profit = reduce(+, _dataframe[!, column])
      record[year][column] = profit
    end
  end

  _years = collect(keys(record))
  _profitmaps = collect(values(record))
  indexs = sortperm(_years)

  years = _years[indexs]
  profitmaps = _profitmaps[indexs]

  p = plot(size = figuresize)
  for column in xs
    profits = map(x -> x[column], profitmaps)
    plot!(p, years, profits, label = column, xticks = :all)
  end

  plot(p) |> display
end

plotCompareProfit(transformedData)
# MODULE 改编电影和原创电影的对比

# DONE 1. 数量对比
function plotCompareOriginal(dataframe::DataFrame)
  column = "is original"
  dataframe[!, column] = map(x -> contains(x, "based on novel") ? 0 : 1, dataframe[!, :keywords])

  keycount = countmap(dataframe[!, column])
  total = keycount[0] + keycount[1]
  xs = ["is original", "not original"]
  ys = [keycount[1] / total, keycount[0] / total]

  pie(xs, ys) |> display
end

plotCompareOriginal(transformedData)
# DONE 2. 平均利润对比
function plotCompareProfit(dataframe::DataFrame)
  column = "is original"
  (notoriginalDataframe, originalDataframe) = groupby(select(dataframe, [column, "profit"]), column)
  # record: Dict{is original, profit}
  originalProfit = reduce(+, originalDataframe[!, :profit])
  notoriginalProfit = reduce(+, notoriginalDataframe[!, :profit])
  originalCount = first(size(originalDataframe))
  notoriginalCount = first(size(notoriginalDataframe))
  bar(xs, [originalProfit / originalCount, notoriginalProfit / notoriginalCount], size = figuresize) |> display
end

plotCompareProfit(transformedData)