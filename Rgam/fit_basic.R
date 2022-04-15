
library(mgcv)
library(ggplot2)
library(visreg)
library(dplyr)
library(GGally)
library(gridExtra)
library(mgcViz)


stations <- read.csv("../DataFit00/model-inputs.csv")
output_dir <- "../DataFit00/"

## 

# Fitting the model takes ...45 s?
# See how it compares to a specific fit for a station
version<-'gam_base'
for( site in unique(stations$site) ) {
  df_site<-stations[ stations$site==site, ]

  mod_base <- bam(ssc_mgL ~ s(usgs_lf) 
                          + s(wind) 
                          + s(tdvel)
                          + s(wl)
                          + s(storm)
                          + s(delta),
                  data=df_site)

  summary(mod_base)
  sink(file.path(output_dir,paste('summary-',site,'-',version,'.txt',sep='')))
  print(summary(mod_base))
  sink()
  
  # So far not looking very convincing.
  # tdvel has a reasonable pattern.
  if( TRUE ) {  
    img_fn<-file.path(output_dir,paste('presid-',site,'-',version,'.png',sep=''))
    png(img_fn,width=9,height=7,res=150,units='in')
    mod<-getViz(mod_base)  
    print(plot(mod,allTerms=T),pages=1)
    dev.off()  
  }
  if ( TRUE ) { # density plot and report RMSE
    pred=predict(mod_base)
    df_site_and_pred<-cbind(df_site, pred)
    
    err<-df_site_and_pred$pred - df_site_and_pred$ssc_mgL
    rmse<-sqrt(mean(err^2))
    
    anno <- data.frame(x=c(-Inf), y=c(Inf),
                       text=c(paste(site,"\nRMSE:",format(rmse,5,3))),
                       hj=c(-0.25), vj=c(2))
    
    img_fn<-file.path(output_dir,paste('density-',site,'-',version,'.png',sep=''))
    png(img_fn,width=9,height=7,res=150,units='in')
    g<-(ggplot(df_site_and_pred,aes(x=ssc_mgL,y=pred)) + 
          stat_density_2d(aes(fill=..level..), geom="polygon", colour="white") +
          #geom_hex(bins=100) +
          geom_text(data=anno,aes(x=x,y=y,label=text,hjust=hj,vjust=vj)))
    print(g)
    dev.off()
  }
  if( FALSE ) {
    panels<-list()
    for ( fld in c('usgs_lf','wind','tdvel','wl','storm','delta') ) {
      panel<-visreg(mod_base,fld,partial=TRUE,gg=TRUE,rug=FALSE) + ylim(-50,200)
      panels<-append(panels,list(panel))
    }
    pan<-grid.arrange(grobs=panels,nrow=1)
    img_fn<-file.path(output_dir,paste('presid-',site,'-',version,'.png',sep=''))
    ggsave(img_fn,plot=pan,width=10,height=3)
  }
  if(FALSE){
    for ( fld in c('usgs_lf','wind','tdvel','wl','storm','delta') ) {
      img_fn<-file.path(output_dir,paste('presid-',site,'-',fld,'-',version,'.png',sep=''))
      png(img_fn,width=9,height=7,res=150,units='in')
      g<-visreg(mod_base,fld,partial=TRUE,gg=TRUE,rug=FALSE) + ylim(-50,200)
      print(g)
      dev.off()
    }  
  }
}

