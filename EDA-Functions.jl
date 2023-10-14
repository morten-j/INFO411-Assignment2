### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 1af951e4-21e0-49fc-b95f-1e4dd90f9986
md"""
###### Functions:
"""

# ╔═╡ bba2fb64-402c-40ba-ab43-2524ca46bbea
md"""
*Separates categorical and continuous columns for plotting distributions:*
"""

# ╔═╡ 53885522-6a4d-11ee-3d42-0b7395737483
function separate(data)

	unique_value_threshold = 10  
	# Identify categorical and continuous columns based on the threshold
	categorical_columns = String[]
	continuous_columns = String[]
	
	for col in names(data)
	    unique_values = unique(data[!, col])
	    if length(unique_values) <= unique_value_threshold
	        push!(categorical_columns, col)
	    else
	        push!(continuous_columns, col)
	    end
	end
	categorical_df = select(data, categorical_columns)
	continuous_df = select(data, continuous_columns)
    return categorical_df , continuous_df
end

# ╔═╡ 8a10064b-e85b-4b0b-80c9-cb095018671e
md"""
*converts float values of categorical columns to integars for bar plots:*
"""

# ╔═╡ 3a9cbc93-5dda-4f50-9435-4a91b780b119
function cat_float_to_int(data)
	for col_name in names(data)
	    # Check if the column contains float values
	    if eltype(data[!, col_name]) <: AbstractFloat
	        # Convert values to integers and assign them back to the column
	        data[!, col_name] .= Int.(data[!, col_name])
		end
	end
	return data
end

# ╔═╡ 123de380-b131-43d3-83bd-9ba834fd3b22
md"""
*Plots histograms and bar plots:*
"""

# ╔═╡ d2ade775-8268-4393-ba79-b60a16f7394c
function hist_bar(data)
	cat1 , con1 = separate(data);
	cbar_plots = [histogram(con1[:,i], xlabel="$(names(con1)[i])", ylabel="amount", legend=false) for i in 1:length(names(con1))] 
	h_p = plot(cbar_plots..., size=(1000, 1000),)
	cat11 = cat_float_to_int(cat1);
	bp = repeat([plot()], length(names(cat11)))
	for col in 1:length(names(cat11))
		s = sort(unique(cat11[:,col]))
		datamap = countmap(cat11[:,col])
		bp[col]=bar((x -> datamap[x]).(s), xticks=(1:length(s), s),xlabel="$(names(cat11)[col])", legend = false)
	end
	b_p = plot(bp..., size=(1000,1000))
	full_plot = plot([b_p,h_p]..., layout=(2, 1), size=(1000, 1000))
end

# ╔═╡ 98a301a5-e270-491c-983f-b7f02ad937fe
md"""
*Calculates correlation matrix and plots it:*
"""

# ╔═╡ 58a9cf6a-3c24-4b66-b09f-47c3c3db8d73
function corr_plot(data)
	correlation_matrix = cor(Matrix(data))
	cmin, cmax = -1.0, 1.0 
	hm = heatmap(correlation_matrix, 
   			xticks=(1:size(correlation_matrix, 1), names(data)),
    		yticks=(1:size(correlation_matrix, 2), names(data)),
	        color=:coolwarm,
	        aspect_ratio=:equal,
			clims= (cmin, cmax),
	        xlabel="Features",
	        ylabel="Features",
	        title="Correlation Matrix", size = (900,900), fontsize= 20)
	for i in 1:size(correlation_matrix, 1)
	    for j in 1:size(correlation_matrix, 2)
	        annotate!(i , j , text(round(correlation_matrix[i, j], digits=2), 10, :white, :center))
    	end
	end
	hm
end

# ╔═╡ b3a12e07-49e9-4610-99f8-e01fa3bd49b9
md"""
*Uses Kmeans clustering and plots it:*
"""

# ╔═╡ cf84aa8f-ad91-4f5f-ac4b-295837ab1555
function clusters(data, col1 , col2)
	attributes_for_clustering = select(data, [col1, col2])
	# Convert the DataFrame to a matrix
	data_matrix = Matrix(attributes_for_clustering)
	k = 2
	# Perform K-means clustering
	result = kmeans(data_matrix', k)
	# Access cluster assignments and centroids
	assignments = result.assignments
	centroids = result.centers
	scatter(data[!,col1], data[!,col2], 
	    group=assignments,  
	    xlabel=col1,
	    ylabel=col2,
	    # zlabel="cp",
	    legend=false,
	    title="K-means Clustering"
	)
end

# ╔═╡ 79458ca4-16d0-4b33-a3bd-a81abc2e2181
md"""
*Converts invalid data types to readable float and missing values:*
"""

# ╔═╡ 5e92bc11-b6a8-44b0-ac67-1d1855550aea
function convert_to_float(raw_data)
	rdata = copy(raw_data)
	# Character to replace
	char_to_replace = "?"
	# Loop through each column and replace the character
	for (colname, coldata) in pairs(eachcol(rdata))
	    if eltype(coldata) <: String || eltype(coldata) <: String1 || eltype(coldata) <: String3 || eltype(coldata) <: String7
	        rdata[!, colname] .= replace.(coldata, char_to_replace => missing)
			rdata[!, colname] .= map(x -> x == "missing" ? missing : parse(Float64, x), rdata[!, colname])    
		end
	end
	return rdata
end

# ╔═╡ a8d60f91-8058-4397-9221-2b581871ea05
md"""
*Checks the number of missing values and discards columns with more than 40% missing values:*
"""

# ╔═╡ a4b3c801-2047-4c20-8d84-6b67549fbb34
function discard_high_missing_cols(fdata)
	count_per_column = [(colname, count(x -> isequal(x, missing), fdata[!, colname])) for colname in names(fdata)]
	QC_fdata = copy(fdata)
	del_cols = []
	df10per = (40*size(QC_fdata)[1])/100
	for (colname, count_value) in count_per_column
		if count_value > df10per
			push!(del_cols, colname)
			select!(QC_fdata, Not(colname))
		end
	end
	return QC_fdata , del_cols
end

# ╔═╡ e4b4cffe-1111-4578-a2f7-896606812490
md"""
*Imputes data using knn:*
"""

# ╔═╡ ddcb06c2-b538-4e94-a6a4-06e03e1253cb
function imputing(qcdata)
	impmat = Impute.knn(
	         convert(Matrix{Union{Missing, Float64}}, Matrix(qcdata)),
	         dims=:cols)
	impmat_float = coalesce.(impmat, 999.999)
	imp_fdf = DataFrame(impmat_float, Symbol.(names(qcdata)))
	return imp_fdf
end

# ╔═╡ 1d63ea6c-ca63-449f-bbce-612fc4fc55d8
md"""
*Standardizes features and creates binary target values:*
"""

# ╔═╡ 702c27b8-fd89-4f02-b412-ef04ddef2e3d
function std_encode(imp_qc_fl, f)
	std_imp_qc_fl = copy(imp_qc_fl)
	if f == 1
		std_imp_qc_fl.num = ifelse.(std_imp_qc_fl.num .== 0, 0, 1)
	end
	unique_thres = 10
	for colname in names(std_imp_qc_fl)
		unique_z = length(unique(std_imp_qc_fl[:,colname]))
		if unique_z < unique_thres
			coerce!(std_imp_qc_fl, :colname => OrderedFactor{unique_z});
		else
			std_imp_qc_fl[:,colname] = standardize(ZScoreTransform, Array(std_imp_qc_fl[:,colname]))
		end
	end
	return std_imp_qc_fl
end

# ╔═╡ 617a2c8c-3b95-47df-a871-4d4837ffe291
md"""
*Runs PCA and plots the dataspace with PC2 and PC2:*
"""

# ╔═╡ 6828e3bf-d7dc-4752-934f-aad407de2c89
function pca_plot(data) 
	coerce!(data, :num => OrderedFactor{2});
    y2, X2 = unpack(data, ==(:num),name->true);
	uniq_labels=unique(y2)
	data = Matrix(X2)
	lb_num=map((x) -> findfirst(==(x), uniq_labels), y2)
	markers=[:ltriangle,:diamond]
	X2t = data'    # or use transpose(data)
	m1 = mean(X2t, dims=2) 
	RS = (X2t .- m1)*(X2t .- m1)'/size(X2t)[2]  
	evals1 = reverse(eigvals(RS))
	evecs1 = reverse(eigvecs(RS), dims = 2)
	proj1 = evecs1[:, 1:2]' * (X2t .- m1)
	p1=scatter()
	for c in uniq_labels
		subs = y2.==c
		scatter!(proj1[1,subs], proj1[2,subs], m=markers[lb_num[subs]], label=c, xlabel="PC1", ylabel="PC2")
	end
	p1
end

# ╔═╡ Cell order:
# ╟─1af951e4-21e0-49fc-b95f-1e4dd90f9986
# ╟─bba2fb64-402c-40ba-ab43-2524ca46bbea
# ╠═53885522-6a4d-11ee-3d42-0b7395737483
# ╟─8a10064b-e85b-4b0b-80c9-cb095018671e
# ╠═3a9cbc93-5dda-4f50-9435-4a91b780b119
# ╟─123de380-b131-43d3-83bd-9ba834fd3b22
# ╠═d2ade775-8268-4393-ba79-b60a16f7394c
# ╟─98a301a5-e270-491c-983f-b7f02ad937fe
# ╠═58a9cf6a-3c24-4b66-b09f-47c3c3db8d73
# ╟─b3a12e07-49e9-4610-99f8-e01fa3bd49b9
# ╠═cf84aa8f-ad91-4f5f-ac4b-295837ab1555
# ╟─79458ca4-16d0-4b33-a3bd-a81abc2e2181
# ╠═5e92bc11-b6a8-44b0-ac67-1d1855550aea
# ╟─a8d60f91-8058-4397-9221-2b581871ea05
# ╠═a4b3c801-2047-4c20-8d84-6b67549fbb34
# ╟─e4b4cffe-1111-4578-a2f7-896606812490
# ╠═ddcb06c2-b538-4e94-a6a4-06e03e1253cb
# ╟─1d63ea6c-ca63-449f-bbce-612fc4fc55d8
# ╠═702c27b8-fd89-4f02-b412-ef04ddef2e3d
# ╟─617a2c8c-3b95-47df-a871-4d4837ffe291
# ╠═6828e3bf-d7dc-4752-934f-aad407de2c89
