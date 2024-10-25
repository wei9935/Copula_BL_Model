import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import py2rpy
import pandas as pd
from finance_data_util import tw_data
from dateutil.relativedelta import relativedelta

# Fit a copula model and similate from it with R, then calculate cov of the simulation
def cop_cov(df, type='RVine'):
    pandas2ri.activate()
    r_df = py2rpy(df)
    r_script = """
    library(quantmod)
    library(VineCopula)
    library(TSP)
    library(fGarch)
    library(network)

    cop_est <- function(x_py, type) {
        prices <- xts(x_py, order.by = as.Date(rownames(x_py), keep.attributes = TRUE))
        loss <- as.data.frame(na.omit(-1.0 * diff(log(prices)) * 100.0)) 
        gfit <- lapply(loss , garchFit , formula = ~ garch(1 ,1), cond.dist = "std", trace = FALSE)
        gprog <- unlist(lapply(gfit , function (x) predict(x, n.ahead = 1)[3]))
        gshape <- unlist(lapply(gfit , function (x) x@fit$coef [5]))
        gresid <- as.matrix(data.frame(lapply(gfit, function (x) x@residuals / sqrt (x@h.t))))
        U <- sapply(1:ncol(loss), function(y) pt(gresid[, y], df = gshape[y]))
        if (type=='DVine'){
            M = 1 - abs(TauMatrix(U))
            #print(M)
            hamilton = insert_dummy(TSP(M),label="cut")
            #print(hamilton)
            sol = solve_TSP(hamilton,method="repetitive_nn")
            #print(sol)
            order = cut_tour(sol,"cut")
            #print(order)
            d <- dim(U)[2]
            #print(d)
            DVM= D2RVine(order,family=rep(0,d*(d-1)/2),par=rep(0,d*(d-1)/2))
            print(DVM)
            vs <- RVineCopSelect(U,familyset=c(1,3,4,5),indeptest=FALSE, level=0.05,
                                Matrix=DVM$Matrix, core=6, selectioncrit="BIC")
        }
        else{
            vs <- RVineStructureSelect(U, familyset=c(1,3,4,5), selectioncrit="BIC",
                                indeptest=FALSE, level=0.05, cores=6, type=type)
        }
        set.seed(123)
        sim <- RVineSim(10000, vs)
        cov <- cov(sim)
        eta <- apply(sim, 2, function(u) qnorm(u))

        return(list(cov, eta))
    }
    """
    robjects.r(r_script)
    robjects.r['options'](warn=-1)
    output = robjects.r['cop_est'](r_df, type)
    cov = output[0]
    cop_residual = output[1]

    return cov, cop_residual

# read json file with series of dataframe, with date as label
def read_cop_cov(file, date):
    if type(file)==str:
        dfs = pd.read_json(file, orient='split')
    else:
        dfs=file
    dfs.index = dfs['Column_Name']
    vine_cov = dfs.loc[date]['Dataframe']

    return pd.DataFrame(vine_cov)

def read_cop_eta(file, date):
    dfs = pd.read_json(file, orient='split')
    dfs.index = dfs['Column_Name']
    vine_res = dfs.loc[date]['Residual']

    return pd.DataFrame(vine_res)




