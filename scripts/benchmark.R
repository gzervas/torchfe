library(fixest)
library(data.table)
library(arrow)

bench <- data.table()

for (f in list.files(path = "data/", full.names = TRUE, recursive = TRUE)) {
    print(f)
    dd <- read_parquet(f)
    fenames <- grep("^fe", colnames(dd), value=T)
    start_time <- Sys.time()
    est <- feols(y ~ x, dd, fixef=fenames, nthreads=0)
    end_time <- Sys.time()
    
    bench <- rbind(
        bench,
        list(
            filename = f,
            n = nrow(dd),
            nfe = length(fenames),
            beta = est$coefficients[1],
            loss = sum(est$residuals**2),
            t_secs = as.double(end_time - start_time),
            t_secs_gpu_xfer = 0
        )
    )
}

fwrite(bench, "output/rbench.csv")
