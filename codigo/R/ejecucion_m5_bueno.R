#########################################################################################
####                              PREPARACIONES PREVIAS                              ####
#########################################################################################

####                              PreparaciOn                                        ####
library(data.table)
library(lightgbm)
library(ggplot2)
library(dplyr)
library(stringr)
library(readr)
library(crayon)


####                              Semillas                                           ####
set.seed(107)

####                              Setear entradas                                    ####
h <- 28 # forecast horizon
max_lags <- 130# number of observations to shift by
first_day <-29 
tr_last <- 1941 # last training day
fday <- as.IDate("2016-05-23") # first day to forecast
nrows <- Inf


####                              Funciones auxiliares                               ####
free <- function() invisible(gc()) #just calls a garbage collector
create_dt_kaggle <- function(is_train = TRUE, nrows = Inf) {
  
  if (is_train) { # create train set
    dt <- fread("Datos/sales_train_evaluation.csv", nrows = nrows,
                drop = paste0("d_", 1:first_day))
    dt=as.data.table(dt)
    cols = colnames(dt)[str_detect(colnames(dt),"d_")]
    dt[, (cols) := transpose(lapply(transpose(.SD),
                                    function(x) {
                                      i <- ifelse(min(which(x>=1))!=Inf,min(which(x>=1)),length(which(x>=1)))
                                      x[1:i-1] <- NA
                                      x})), .SDcols = cols]
    free()
  } else { # create test set
    dt <- fread("Datos/sales_train_evaluation.csv", nrows = nrows,
                drop = paste0("d_", 1:first_day)) # keep only max_lags days from the train set
    dt[, paste0("d_", (tr_last+1):(tr_last+2*h)) := 0] # add empty columns for forecasting
  }
  
  dt <- na.omit(melt(dt,
                     measure.vars = patterns("^d_"),
                     variable.name = "d",
                     value.name = "sales"))
  
  cal <- fread("Datos/calendar.csv")
  cal$event_name_1_dia_despues=c(" ",cal$event_name_1[1:(nrow(cal)-1)])
  dt <- dt[cal, `:=`(date = as.IDate(i.date, format="%Y-%m-%d"), # merge tables by reference
                     wm_yr_wk = i.wm_yr_wk,
                     event_name_1 = i.event_name_1,
                     event_name_2 = i.event_name_2,
                     event_type_1 = i.event_type_1,
                     event_type_2 = i.event_type_2,
                     event_name_1_dia_despues=i.event_name_1_dia_despues,
                     snap_CA = i.snap_CA,
                     snap_TX = i.snap_TX,
                     snap_WI = i.snap_WI), on = "d"]
  
  prices <- fread("Datos/sell_prices.csv")
  dt[prices, sell_price := i.sell_price, on = c("store_id", "item_id", "wm_yr_wk")] # merge again
  
  dt[, `:=`(wday = wday(date), # time features
            mday = mday(date),
            week = week(date),
            month = month(date),
            quarter=quarter(date),
            year = year(date))]
  
  
  cols <-
    c(
      "item_id",
      "store_id",
      "state_id",
      "dept_id",
      "cat_id",
      "event_name_1",
      "event_name_2",
      "event_type_1",
      "event_type_2",
      "event_name_1_dia_despues",
      "wday",
      "mday",
      "week",
      "month",
      "quarter",
      "year"
    )
  free()
  dt[, (cols) := lapply(.SD, function(x) as.integer(factor(x))), .SDcols = cols] # convert character columns to integer
  
  if (!is_train){
    ind=which(as.numeric(str_remove_all(dt$d,"d_"))>=(tr_last-max_lags))
    dt=dt[ind,]
  }
  return(dt)
  
}#creates a training or testing data table from a wide-format file with leading zeros removed.
create_fea <- function(dt) {
  dt[, `:=`(d = NULL, # remove useless columns
            wm_yr_wk = NULL)]
  
  lag <- c(28,56)
  lag_cols <- paste0("lag_", lag) # lag columns names
  dt[, (lag_cols) := shift(.SD, lag), by = id, .SDcols = "sales"] # add lag vectors
  # 
  win <- c(28,56) # rolling window size
  roll_cols <- paste0("rmean_", t(outer(lag, win, paste, sep="_"))) # rolling features columns names
  dt[, (roll_cols) := frollmean(.SD, win, na.rm = TRUE), by = id, .SDcols = lag_cols] # rolling features on lag_cols
  # 
  dt=dt%>%mutate(caro=ifelse(sell_price>10,1,0))
  # dt=dt%>%mutate(experimento_1=sell_price/rmean_7_28)
  # dt=dt%>%mutate(experimento_2=(rmean_7_7-rmean_7_28)/lag_7)
  # lag <- c(56,112)
  # lag_cols <- paste0("lag_", lag) # lag columns names
  # dt[, (lag_cols) := shift(.SD, lag), by = id, .SDcols = "sales"] # add lag vectors
  # 
  # win <- c(56,112) # rolling window size
  # roll_cols <- paste0("rmean_", t(outer(lag, win, paste, sep="_"))) # rolling features columns names
  # dt[, (roll_cols) := frollsum(.SD, win, na.rm = TRUE), by = id, .SDcols = lag_cols] # rolling features on lag_cols
  # 
  # dt[, `:=`(lag_14 = NULL, # remove useless columns
  #           lag_56 = NULL,
  #           lag_730 = NULL)]
  
  
  
} #adds lags, rolling features and time variables to the data table
training<-function(cop,te,variable=var,variable_now=as.integer(phase[i]),iter,p){
  #########################################################################################
  ####                              ENTRENAMOS DATOS SIN EVENTOS SOLO                  ####
  #########################################################################################
  set.seed(round(runif(1,1,100)))
  tr = cop[which(cop%>%select(variable)==variable_now),]
  tr=na.omit(tr)#%>% filter(event_name_1_i == 1 & event_name_2_i == 1)
  # tr=tr%>%mutate(finde=ifelse(wday>2,0,1))
  # tr=tr%>%mutate(festivo=ifelse(event_name_1>1,1,0))
  
  tr = as.data.table(tr)
  idx = runif(round(0.05 * nrow(tr)), 1, nrow(tr))
  # idx = which(tr$date>="2015-04-24" & tr$date<="2015-05-21")
  y <- tr$sales*tr$sell_price
  
  #Eliminamos columnas inservibles
  tr[, c(
    "id",
    "sales",
    "date",
    # "event_name_1",
    # "event_name_1_dia_despues",
    "event_name_2",
    # "event_type_1",
    "event_type_2",
    variable
  ) := NULL]
  
  free()
  tr <- data.matrix(tr)
  free()
  
  #Definimos columnas categ?ricas
  cats <-
    c("item_id", "store_id", "state_id", "dept_id", "cat_id",
      "event_name_1",
      # "wday",
      # "mday",
      # "week",
      # "month",
      "quarter",
      # "year"
      "event_name_1_dia_despues",
      # "event_name_2"
      "event_type_1"
      # "event_type_2"
    ) # list of categorical features
  cats=setdiff(cats,variable)
  xtr <-lgb.Dataset(tr[-idx,], label = y[-idx], categorical_feature = cats) # construct lgb dataset
  xval <-lgb.Dataset(tr[idx,], label = y[idx], categorical_feature = cats)
  rm(tr, y, cats, idx)
  free()
  
  #Entrenamos modelo sin eventos
  m_lgb <- lgb.train(
    params = p,
    data = xtr,
    nrounds = iter,
    valids = list(val = xval),
    early_stopping_rounds = 20,
    eval_freq = 20
  )
  i=paste0(variable,"_",variable_now)
  lgb.save(m_lgb, paste0(variable,"_",variable_now,".txt"))
  cat("Best score:",
      m_lgb$best_score,
      "at",
      m_lgb$best_iter,
      "iteration")
  free()
  
  rm(xtr, xval)
  
  #Guardamos gr?fica de importancia de variables
  imp <- lgb.importance(m_lgb)
  
  a = imp[order(-Gain)][1:35, ggplot(.SD, aes(reorder(Feature, Gain), Gain)) +
                          geom_col(fill = "steelblue") +
                          xlab("Feature") +
                          coord_flip() +
                          theme_minimal()]
  a
  
  # 
  ggsave( paste0(variable,"_",variable_now,".png"))
  # 
  #Predecimos ventas para dias sin eventos
  # te_sin = te
  #m_lgb <- lgb.load(filename = paste0(variable,"_",variable_now,".txt"))
  if (variable=="event_name_1"){
    days=unique(te[which(te%>%select(variable)==variable_now),"date"])
    days=days[days$date>=fday & days$date<=fday+27]$date
  }else{
    days=as.list(seq(fday, length.out = h, by = "day"))
  }
  j=2
  for (day in days) {
    cat(as.character(day), " ")
    if (variable=="event_name_1"){
      te_1=te#[which(te%>%select(variable)==variable_now),]
    }else{
      te_1=te[which(te%>%select(variable)==variable_now),]
    }
    
    tst <- te_1[date >= day - max_lags & date <= day ]
    create_fea(tst)
    tst <-
      data.matrix(tst[date == day][, c(
        "id",
        "sales",
        "date",
        # "event_name_1",
        # "event_name_1_dia_despues",
        "event_name_2",
        # "event_type_1",
        "event_type_2",
        variable
        # "event_name_1_i",
        # "event_name_2_i",
        # "event_type_1_i",
        # "event_type_2_i"
      ) := NULL])
    te[intersect(which(te$date == day),which(te%>%select(variable)==variable_now)), sales := predict(m_lgb, tst)/tst[,"sell_price"]]
    # print(paste0("Mio: ",mean(te[date == day, sales]),"  Mejor: ",2*mean(mejor[,j,drop=TRUE],na.rm = T)))
    j=j+1
  }
  return(te)
}
####                              Preparamos training                                ####
setwd("C:/Users/PACO/Google Drive (paco.noa.gut@gmail.com)/ARC_KAGGLE/m5/Proyect")
tr <- create_dt_kaggle()
free()
create_fea(tr)
free()
tr <- na.omit(tr) # remove rows with NA to save memory
free()
cop=tr

####                              Par?metros modelos                                 ####
p <- list(objective = "poisson",
          metric ="rmse",
          force_row_wise = TRUE,
          learning_rate = 0.085,
          num_leaves = 120,
          min_data = 100,
          # sub_feature = 0.75,
          sub_row = 0.72,
          bagging_freq = 1,
          lambda_l2 = 0.09,
          nthread = 12)





library(readr)
cat_idprueba_1 <- read_csv("cat_id_new_variables/cat_id_new_variables_Prueba2/cat_idprueba_1.csv")
e=as.vector(apply(cat_idprueba_1[,2:29],2,mean))


ggplot()+
  geom_line(aes(x=1:28,y
for (kk in 1:20){
  p <- list(objective = "poisson",
            metric ="rmse",
            force_row_wise = TRUE,
            learning_rate = runif(1,0.025,0.195),
            num_leaves = round(runif(1,50,230)),
            min_data =round(runif(1,40,360)),
            # sub_feature = 0.75,
            sub_row = runif(1,0.23,0.97),
            bagging_freq = 1,
            lambda_l2 = runif(1,0.005,0.22),
            nthread = 12)
  saveRDS(p,paste0(kk,".rds"))
  
  print(p)
  var="cat_id"
  phase=unique(cop%>%select(var))
  i=1
  
  
  #mejor <- read_csv("C:/Users/PACO/Google Drive (paco.noa.gut@gmail.com)/ARC_KAGGLE/m5/dats_subs_final/mejor.csv")
  # mejor <- read_csv("C:/Users/PACO/Google Drive (paco.noa.gut@gmail.com)/ARC_KAGGLE/m5/dats_subs_final/submission_Darker_Magic.csv")
  
  
  # Directorio_Paco="C:/Users/PACO/Google Drive (paco.noa.gut@gmail.com)/ARC_KAGGLE/m5/dats_subs_final"
  # setwd(Directorio_Paco)
  setwd("C:/Users/PACO/Google Drive (paco.noa.gut@gmail.com)/ARC_KAGGLE/m5/Proyecto_M5")
  free()
  te <- create_dt_kaggle(FALSE, nrows)
  vari=paste0(var,"_new_variables")
  dir.create(vari)
  setwd(paste0(getwd(),"/",vari))
  num=list.files()
  dir.create(paste0(vari,"_Prueba",length(num)+1))
  dir_guard=paste0(getwd(),"/",vari,"_Prueba",length(num)+1)
  setwd(dir_guard)
  # te$cat_state=paste0(te$cat_id,te$state_id)
  #unique(te$event_name_1)
  for (i in 1:nrow(phase)){
    cat(green(paste0("Ejecutanto el ",vari, ":")))
    cat(green(paste0("\n",i)))
    te=training(cop,te,variable=var,variable_now=as.integer(phase[i]),iter=1450,p)
  }
  
  te[date >= fday
     ][date >= fday+h, id := sub("validation", "evaluation", id)
       ][, d := paste0("F", 1:28), by = id
         ][, dcast(.SD, id ~ d, value.var = "sales")
           ][, fwrite(.SD,paste0(var, "prueba_1.csv"))]
  
  
  
}=e))
  
  
  