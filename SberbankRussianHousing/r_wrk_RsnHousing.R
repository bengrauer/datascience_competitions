# RECENTLY ADDED - Added macro - economics
# Set the working directory
setwd('/Project/data/kg_RussiaHousing')

# Clean out
rm(list=ls(all=TRUE)) 

library(caret)
library(dplyr)
library(lubridate)
library(ggplot2)
require(data.table)

library(data.table)
library(Matrix)
library(caret)

# library(sqldf)
# library(zoo)
# library(xgboost)

# Train + Macro - load
trainDT <- fread("train.csv")
macroDT <- fread("macro.csv")
testDT <- fread("test.csv")
head(trainDT)
head(macroDT)
tail(macroDT)


  # Transform time-stamp for macro level factors
  macroDT$NewTimestamp <- as.Date(macroDT$timestamp, format="%m/%d/%Y")
  macroDT$NewTimestamp <- as.character(macroDT$NewTimestamp + years(2000))
  summary(macroDT$timestamp)
  summary(macroDT$NewTimestamp)
  # replace original
  macroDT$timestamp <- macroDT$NewTimestamp
  
  # Merge the two
  fullDT <- merge(trainDT, macroDT, by.x = "timestamp", by.y = "timestamp")
  head(fullDT)
  
  # join the train + test
  trainDT <- merge(trainDT, macroDT, by.x = "timestamp", by.y = "timestamp")
  testDT <- merge(testDT, macroDT, by.x = "timestamp", by.y = "timestamp")



# Check the counts
count(trainDT) # 30471 obs
count(macroDT)    # 2484 obs
count(fullDT)     # 30741 obs





########################################
# ADD SOME BASIC FEATURES TO CLEAN WITH
########################################

# IF the life_sq is null, populate with full_sq.  Maybe convert this later
# trainDT[trainDT$life_sq == is.na(trainDT$life_sq)]

################
# DATA QUALITY
################


# First need to address full_sq.  It has 0's in it.  Take the average sq_ft per each sub-category
clean.trainDT <- trainDT
clean.testDT <- testDT

# Look at items that have "0" value amounts
filter(clean.trainDT, full_sq == 0)
clean.trainDT[clean.trainDT$price_doc ==0]


# AVERAGE full_sq FOR each SUB-AREA, then join this 
# Grab the average sub area - full sq
  avgSubAreaFullSq_Train <-
    clean.trainDT %>% 
      group_by(sub_area) %>%
      summarise(ft_avgSubAreaSize = mean(full_sq))
  
  avgSubAreaFullSq_Train <- as.data.frame(avgSubAreaFullSq_Train)
  
  # Also do for test
  avgSubAreaFullSq_Test <-
    clean.testDT %>% 
    group_by(sub_area) %>%
    summarise(ft_avgSubAreaSize = mean(full_sq))
  
  avgSubAreaFullSq_Test <- as.data.frame(avgSubAreaFullSq_Test)
  
  # Join the data sets on the average
  # TODO: Need to work the merge a little bit more.  Taking a col and join and putting all the data at the far left
  clean.trainDT <- merge(clean.trainDT, avgSubAreaFullSq_Train, by="sub_area" )
  clean.testDT <- merge(clean.testDT, avgSubAreaFullSq_Test, by="sub_area")
  
  head(clean.trainDT)
  # Set the full_sq = to avg are of sub-area, for those that are 0
  clean.trainDT[clean.trainDT$full_sq==0]$full_sq <- as.integer(round(clean.trainDT$ft_avgSubAreaSize, digits = 0))
  clean.testDT[clean.testDT$full_sq==0]$full_sq <- as.integer(round(clean.testDT$ft_avgSubAreaSize, digits=0))  # TODO: Need to handle for NA
  # Review items
  summary(trainDT$full_sq)
  summary(clean.trainDT$full_sq)
  summary(clean.trainDT$ft_avgSubAreaSize)
  summary(clean.testDT$full_sq)
  
  # check na - good
  sum(is.na(clean.trainDT$full_sq))
  sum(is.na(clean.testDT$full_sq))



# TODO: IF TIME PERMITS - NEED TO FIGURE OUT WHAT TO DO WITH THIS LATER 
# SQFT PER ROOM.  num_room has too many NA
  clean.trainDT$ft_sqrftPerRoom <- clean.trainDT$full_sq / clean.trainDT$num_room
  clean.testDT$ft_sqrftPerRoom <- clean.testDT$full_sq / clean.testDT$num_room
  
  summary(clean.trainDT$ft_sqrftPerRoom)
  summary(clean.testDT$ft_sqrftPerRoom)
  
  # Clear out infinity TODO: Need to change this to averages
  clean.trainDT$ft_sqrftPerRoom <- replace(clean.trainDT$ft_sqrftPerRoom, is.infinite(clean.trainDT$ft_sqrftPerRoom), 0)
  clean.trainDT$ft_sqrftPerRoom <- replace(clean.trainDT$ft_sqrftPerRoom, is.na(clean.trainDT$ft_sqrftPerRoom), 0)
  
  clean.testDT$ft_sqrftPerRoom <- replace(clean.testDT$ft_sqrftPerRoom, is.infinite(clean.testDT$ft_sqrftPerRoom), 0)
  clean.testDT$ft_sqrftPerRoom <- replace(clean.testDT$ft_sqrftPerRoom, is.na(clean.testDT$ft_sqrftPerRoom), 0)

  # TODO: Think about the price per sq foot for those populated and then reverse apply from there as a feature

  # clean this high level - data anomoly.  4000 sq for a super low end price?
  # TODO: if time permits, changes this into looking at the price / sq foot ratio instead of a hard #
  # trainDT[trainDT$full_sq > 4000]  # ID = 3530
  clean.trainDT <- clean.trainDT[clean.trainDT$full_sq < 4000]

  # LEAVE OUT
  # trainDT[trainDT$num_room > 10]  # ID = 11624, 26716.  If the number of rooms / square foot, doesn't make sense, get rid of this data
  # clean this with a square footage calculation - KEEP THIS OUT THIS MADE THE SCORE WORSE
  # clean.trainDT <- clean.trainDT[!clean.trainDT$ft_sqrftPerRoom < 10]

# TRY - floor, maxfloor, product_type
  # Perhaps include the floor + maxfloor
  # Set the floor = 1 where missing
  clean.trainDT$floor <- replace(clean.trainDT$floor, is.na(clean.trainDT$floor), 0)
  clean.testDT$floor <- replace(clean.testDT$floor, is.na(clean.testDT$floor), 0)
  
  table(clean.trainDT$floor)
  table(clean.testDT$floor)
  
  # Also try to PREDICT FOR EACH AREA

# NUMBER OF ROOMS
  # Let's do num_room, ttk_km - they seem to be correlated to price
  summary(clean.trainDT$num_room)
  # replace nas and 0 with 1
  clean.trainDT$num_room <- replace(clean.trainDT$num_room, is.na(clean.trainDT$num_room), 1)
  clean.trainDT$num_room[clean.trainDT$num_room == 0] <- 1
  
  clean.testDT$num_room <- replace(clean.testDT$num_room, is.na(clean.testDT$num_room), 1)
  clean.testDT$num_room[clean.testDT$num_room == 0] <- 1
  
  table(clean.trainDT$num_room)
  table(clean.testDT$num_room)
  
  # now add a studio or bedrooms - TODO: Move down to features
  clean.trainDT$ft_livingtype <- ifelse(clean.trainDT$num_room==1, "Studio", "Rooms")
  clean.testDT$ft_livingtype <- ifelse(clean.testDT$num_room==1, "Studio", "Rooms")
  
  # already clean
  summary(clean.trainDT$ttk_km)
  
#########################################################
# Kitchen sq - Delete anything where the kitchen_sq > full_sq
  count(clean.trainDT) # 22852
  # TODO: Here is the bad data for kitch_sq, we just need to fix it
  clean.trainDT[(clean.trainDT$kitch_sq > clean.trainDT$full_sq)]
  clean.trainDT[clean.trainDT$kitch_sq > 100]  
  
  # Here set any entries where the kitchen_sql >= full_sq to NA.
  count(clean.trainDT) # 22852
  clean.trainDT[(clean.trainDT$kitch_sq >= clean.trainDT$full_sq)]$kitch_sq <- NA
  clean.testDT[(clean.testDT$kitch_sq >= clean.testDT$full_sq)]$kitch_sq <- NA
  
  # Perhaps get the average sq/ft per kitchen for each sub_area and impute with that.  Or size it as small/medium/large
  avgSubAreaKitchSq_Train <-
    clean.trainDT %>% 
    group_by(sub_area) %>%
    summarise(ft_KitchSq_avg = as.integer(round(mean(kitch_sq, na.rm=TRUE), digits=0)))
  
  avgSubAreaKitchSq_Train <- as.data.frame(avgSubAreaKitchSq_Train)
  
  table(clean.trainDT$sub_area, clean.trainDT$kitch_sq)
  
  # Also do for test
  avgSubAreaKitchSq_Test <-
    clean.testDT %>% 
    group_by(sub_area) %>%
    summarise(ft_KitchSq_avg = as.integer(round(mean(kitch_sq, na.rm=TRUE), digits=0)))
  
  avgSubAreaKitchSq_Test <- as.data.frame(avgSubAreaKitchSq_Test)
  
  # Merge
  clean.trainDT <- merge(clean.trainDT, avgSubAreaKitchSq_Train, by="sub_area" )
  clean.testDT <- merge(clean.testDT, avgSubAreaKitchSq_Test, by="sub_area")
  
  # replace nas and 0 with 1
  # Clean 0
  clean.trainDT[clean.trainDT$kitch_sq==0]$kitch_sq <- clean.trainDT[clean.trainDT$kitch_sq==0]$ft_KitchSq_avg
  clean.testDT[clean.testDT$full_sq==0]$kitch_sq <- clean.testDT[clean.testDT$kitch_sq==0]$ft_KitchSq_avg  # TODO: Need to handle for NA
  # Clean NA
  clean.trainDT[is.na(clean.trainDT$kitch_sq)]$kitch_sq <- clean.trainDT[is.na(clean.trainDT$kitch_sq)]$ft_KitchSq_avg
  clean.testDT[is.na(clean.testDT$kitch_sq)]$kitch_sq <- clean.testDT[is.na(clean.testDT$kitch_sq)]$ft_KitchSq_avg
  # Check Data
  summary(trainDT$kitch_sq)
  summary(clean.trainDT$kitch_sq)
  summary(clean.trainDT$ft_avgSubAreaSize)


# Test each of our features across train/test
summary(trainDT$price_doc)
summary(trainDT$ft_sqrftPerRoom)
summary(testDT$ft_sqrftPerRoom)
head(table(testDT$ft_sqrftPerRoom), 10)


####################
# FEATURE ADD
####################

  # Add Year to the mix - don't think I can use.  train has 2013-2015.  test has 2015/2016.  Makes sense I can't use
  clean.trainDT$year <- year(clean.trainDT$timestamp)
  clean.testDT$year <- year(clean.testDT$timestamp)

  table(clean.trainDT$year)
  table(clean.testDT$year)
  
# MAX FLOOR
  # Anything above ground 3 might be nice
  summary(clean.trainDT$max_floor)
  summary(clean.trainDT$floor)
  
  clean.trainDT$max_floor <- ifelse(is.na(clean.trainDT$max_floor), as.integer(clean.trainDT$floor), as.integer(clean.trainDT$max_floor))
  clean.testDT$max_floor <- ifelse(is.na(clean.testDT$max_floor), as.integer(clean.testDT$floor), as.integer(clean.testDT$max_floor))
  
  clean.trainDT$ft_floorType <- ifelse(clean.trainDT$num_room<=2, "Ground", "HighRise")
  clean.testDT$ft_floorType <- ifelse(clean.testDT$num_room<=2, "Ground", "HighRise")

# KITCHEN SIZE
  # Here take a look at the kitchen size and how it is distributed (in plot section).  I'm looking to make a small/vs large kitchen size feature.
  # There are more past 20, but it tails off
  clean.trainDT$ft_KitchSq_SizeType <- ifelse(clean.trainDT$kitch_sq <= 4, "Small", "Large")
  clean.testDT$ft_KitchSq_SizeType <- ifelse(clean.testDT$kitch_sq <= 4, "Small", "Large")
  
# MONTH - Spring and Summer Months are hotter markets?  At least in America, may not be the case over-seas.  Eric said his place charges higher rates at peak seasons
  clean.trainDT$ft_month <- month(clean.trainDT$timestamp)
  clean.testDT$ft_month <- month(clean.testDT$timestamp)
  
  
####################
# ADDITIONAL PLOTS
####################
# If we want to show plots (set to 1) to bypass with running of script
bShowPlots = 0
if (bShowPlots == 1)
  {
    
    # Anomaly - there is a full_sq out of range for the price - possibly delete this item.
    qplot(price_doc, full_sq, data=fullDT, color=state)
    head(sort(trainDT$full_sq, decreasing = TRUE))
  
    # Kitchen size after cleaning
    ggplot(clean.trainDT, aes(x=kitch_sq)) + geom_histogram(aes(y=..density..), binwidth=.5, colour="black", fill="white") +
      geom_density(alpha=.2, fill="#FF6666") + scale_x_continuous(name="sq ft", limits=c(0,20), breaks=seq(0,20,1))
    
    # rooms more than 15??, and still cheap??  these looks like anomolys.  Check their square footage
    qplot(price_doc, full_sq, data=fullDT, color=state)
    qplot(price_doc, num_room, data=fullDT, color=state)
    qplot(price_doc, num_room, data=trainDT, color=state)
    # qplot(price_doc, ft_sqrftPerRoom, data=trainDT, color=state) - return if we get square foot per room sorted out for nulls
    # qplot(price_doc, full_sq, data=trainDT, color=sub_area) # Too many entries here
    
    qplot(clean.trainDT$full_sq, clean.trainDT$price_doc, color=clean.trainDT$state)
    qplot(price_doc, full_sq, data=clean.trainDT, color=state)
    
    # ggplot
    qplot(clean.trainDT$full_sq, clean.trainDT$price_doc)
    
    # Trying some different plot types to re-use for example
    plot(clean.trainDT$full_sq, clean.trainDT$price_doc)
    abline(lm.PrcPrFullSqFt, col="red")
    
    ggplot(clean.trainDT, aes(x = full_sq, y = price_doc)) + 
      geom_point() + stat_smooth(method = "lm", col = "red")
    
    ggplot(clean.trainDT, aes(x = full_sq, y = price_doc), color=sub_area) + 
      geom_point() + stat_smooth(method = "lm", col = "red")
    
    # histogram of state vs price doc to see if others are higher
    ggplot(clean.trainDT, aes(x = as.factor(state), y = price_doc)) +
      geom_boxplot() 
  }
  

#######################
# SPLIT OUT THINGS HERE - Will work this later
#######################
# Splitting for further testing

  # Un-scrubbed data
  # trainDT, testDT
  
  # Original Cleaned Data
  orig_clean.trainDT <- clean.trainDT
  orig_clean.testDT <- clean.testDT
  
  # If we are running a local test - split the data 75/25
  # 0 = local split
  # 1 = prod submission
  bRunLocalTest <- 0
  
  # Split the data
  if( bRunLocalTest == 1 )
  {
  
    smp_size <- floor(0.75 * nrow(clean.trainDT))
    set.seed(123)
    train_ind <- sample(seq_len(nrow(clean.trainDT)), size=smp_size)
    
    clean.trainDT <- orig_clean.trainDT[train_ind]
    clean.testDT <- orig_clean.trainDT[-train_ind]
  }else
    # If i want to revert without re-running the entire script
  {
    clean.trainDT <- orig_clean.trainDT
    clean.testDT <- orig_clean.testDT
  }
  

  
  
#####################
# LINEAR MODEL
#####################
  
bRunStdLin <- 0
  if (bRunStdLin==1)
  {
  
    # Full Test
    lm1 <- lm(price_doc ~ full_sq + as.factor(ft_livingtype) + as.factor(ft_floorType) + as.factor(ft_KitchSq_SizeType) + as.factor(ft_month) + 
                as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000) + as.factor(church_count_500), data=clean.trainDT,na.action=na.exclude)
    
    # Local Test - May not fit all levels - had to remove church
    if(bRunLocalTest==1)
    {    
      lm1 <- lm(price_doc ~ full_sq + as.factor(ft_livingtype) + as.factor(ft_floorType) + as.factor(ft_KitchSq_SizeType) + as.factor(ft_month) + 
                  as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000) , data=clean.trainDT,na.action=na.exclude)    
      mean((clean.testDT$price_doc - predict.lm(lm1, clean.testDT)) ^ 2)  
    }

    # Review    
    summary(lm1)
    
    pred1 <- predict(lm1, newdata = clean.testDT, interval = "prediction")
    
    # Bind and submit
    final.testDT <- cbind(clean.testDT, pred1)
    final.testDT <- final.testDT[,c("id","fit")]
    final.testDT <- setNames(final.testDT, c("id","price_doc"))
    final.testDT <- final.testDT[order(id),]
  
    # Check the final outcome
    summary(final.testDT)
    final.testDT[final.testDT$price_doc < 0]
    # final.testDT$price_doc[final.testDT$price_doc < 0] <- mean(final.testDT$price_doc) - shouldn't have to use - just expirementing to see final rate
    
    # Export 
    write.csv(final.testDT,file="submission.csv",row.names = F)
    
    
    #################################
    # PREDICTIONS - 75% SPLIT
    #################################
    
    # lm1 <- lm(price_doc ~ full_sq, data=clean.trainDT,na.action=na.exclude)
    # 1.330227e+13
    
    # lm1 <- lm(price_doc ~ full_sq + floor, data=clean.trainDT,na.action=na.exclude)
    # 1.33404e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(year), data=clean.trainDT,na.action=na.exclude)
    # 1.339264e
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(year) + as.factor(university_top_20_raion), data=clean.trainDT,na.action=na.exclude)
    # 1.267263e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(university_top_20_raion), data=clean.trainDT,na.action=na.exclude)
    # 1.262933e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(university_top_20_raion) + as.factor(church_count_500), data=clean.trainDT,na.action=na.exclude)
    # 1.235847e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(university_top_20_raion) + as.factor(church_count_500) + as.factor(sport_count_500), data=clean.trainDT,na.action=na.exclude)
    # 1.188449e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(university_top_20_raion) + as.factor(church_count_500) + as.factor(sport_count_500) + as.factor(product_type), data=clean.trainDT,na.action=na.exclude)
    # 1.16679e+13
    # if we do a summary - you can see teh church having a negative impact pushing items down to zero
    
    # Here taking out the church took the negatives out of the picture
    # lm1 <- lm(price_doc ~ full_sq + as.factor(university_top_20_raion) + as.factor(sport_count_500) + as.factor(product_type), data=clean.trainDT,na.action=na.exclude)
    # 1.173131e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(university_top_20_raion) + as.factor(church_count_500) + as.factor(sport_count_500) + factor(ft_livingtype), data=clean.trainDT,na.action=na.exclude)
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(product_type) + factor(ft_livingtype) + as.factor(university_top_20_raion) +  as.factor(sport_count_500), data=clean.trainDT,na.action=na.exclude)
    # 1.166322e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(product_type) + factor(ft_livingtype) + 
    # as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000), data=clean.trainDT,na.action=na.exclude)
    # 1.139335e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(product_type) + factor(ft_livingtype) + factor(ft_floorType) + 
    # as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000), data=clean.trainDT,na.action=na.exclude)
    # 1.131721e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(product_type) + factor(ft_livingtype) + factor(ft_floorType) + as.factor(ft_month) + 
    # as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000), data=clean.trainDT,na.action=na.exclude)
    # 1.132054e+13 - higher than before
    
    
    #################################
    # PREDICTIONS - RAW PROD SET
    #################################
    
    # lm1 <- lm(price_doc ~ full_sq, data=clean.trainDT,na.action=na.exclude)
    # 1.330227e+13
    
    # lm1 <- lm(price_doc ~ full_sq + floor, data=clean.trainDT,na.action=na.exclude)
    # 1.33404e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(year), data=clean.trainDT,na.action=na.exclude)
    # 1.339264e
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(year) + as.factor(university_top_20_raion), data=clean.trainDT,na.action=na.exclude)
    # 1.267263e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(university_top_20_raion), data=clean.trainDT,na.action=na.exclude)
    # 1.262933e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(university_top_20_raion) + as.factor(church_count_500), data=clean.trainDT,na.action=na.exclude)
    # 1.235847e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(university_top_20_raion) + as.factor(church_count_500) + as.factor(sport_count_500), data=clean.trainDT,na.action=na.exclude)
    # 1.188449e+13
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(university_top_20_raion) + as.factor(church_count_500) + as.factor(sport_count_500) + as.factor(product_type), data=clean.trainDT,na.action=na.exclude)
    # 1.16679e+13
    # if we do a summary - you can see teh church having a negative impact pushing items down to zero
    
    # Here taking out the church took the negatives out of the picture
    # lm1 <- lm(price_doc ~ full_sq + as.factor(university_top_20_raion) + as.factor(sport_count_500) + as.factor(product_type), data=clean.trainDT,na.action=na.exclude)
    # 1.173131e+13
    
    # linear model for price per full sq foot
    # 41177
    # modelFit <- train(price_doc ~ full_sq, data=clean.trainDT, method="glm")
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(university_top_20_raion) + as.factor(church_count_500) + as.factor(sport_count_500), data=clean.trainDT,na.action=na.exclude)
    # Kaggle Score - 0.39999
    
    # took out "+ as.factor(product_type) " as it was causing NAs?
    # lm1 <- lm(price_doc ~ full_sq + as.factor(ft_livingtype) + as.factor(ft_floorType) + 
    #            as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000), data=clean.trainDT,na.action=na.exclude)
    # Kaggle Score - 0.39624 
    
    # lm1 <- lm(price_doc ~ full_sq + as.factor(ft_livingtype) + as.factor(ft_floorType) + 
    #            as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000) + as.factor(church_count_500), data=clean.trainDT,na.action=na.exclude)
    # kaggle Score - 0.39802,
    
    # kaggle Score - 0.39647
    # lm1 <- lm(price_doc ~ full_sq + as.factor(ft_livingtype) + as.factor(ft_floorType) + as.factor(ft_KitchSq_SizeType) +
    #         as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000) + as.factor(church_count_500), data=clean.trainDT,na.action=na.exclude)
    
    #   lm1 <- lm(price_doc ~ full_sq + as.factor(ft_livingtype) + as.factor(ft_floorType) + as.factor(ft_KitchSq_SizeType) + as.factor(ft_month) + 
    #       as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000) + as.factor(church_count_500), data=clean.trainDT,na.action=na.exclude)  
    # Kaggle Score - 0.39728    
    
  }
  
  
  
###############
# Polynomial
###############
bRunPoly <- 1
if (bRunPoly==1)
{
  
  # Local SPlit
  if(bRunLocalTest==1)
  {
    
    modelPoly <- lm(price_doc ~ poly(full_sq, 4) + as.factor(product_type) + factor(ft_livingtype) + factor(ft_floorType) + as.factor(ft_month) + 
                      as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000) + as.factor(church_count_500)
                    , data=clean.trainDT)
    mean((clean.testDT$price_doc - predict.lm(modelPoly, clean.testDT)) ^ 2)  
    
  }
  
  # Production form here on down
  modelPoly <- lm(price_doc ~ poly(full_sq, 5) + as.factor(ft_livingtype) + as.factor(ft_floorType)  +
                  as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000), data=clean.trainDT,na.action=na.exclude)
  predPoly <- as.data.frame(predict(modelPoly, clean.testDT))
  
  # Take predictions and load
  predPoly <- setnames(predPoly,c("fit"))
  final.testDT <- cbind(clean.testDT, predPoly)
  final.testDT <- final.testDT[,c("id","fit")]
  final.testDT <- setNames(final.testDT, c("id","price_doc"))
  final.testDT <- final.testDT[order(id),]
  
  # Final Check
  summary(final.testDT$price_doc) 
  final.testDT[final.testDT$price_doc < 0]
  
  write.csv(final.testDT,file="submission.csv",row.names = F)
  
  ##############################
  # LOCAL TEST SPLIT FROM TRAIN
  ##############################
  # With polynomial
  # model5 <- lm(price_doc ~ poly(full_sq, 3) + as.factor(product_type) + factor(ft_livingtype) + factor(ft_floorType) + as.factor(ft_month) + 
  #  as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000), data=clean.trainDT)
  # 1.089728e+13
  
  # model5 <- lm(price_doc ~ poly(full_sq, 4) + as.factor(product_type) + factor(ft_livingtype) + factor(ft_floorType) + as.factor(ft_month) + 
  # as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000), data=clean.trainDT)
  # pred5 <- as.data.frame(predict(model5, clean.testDT))
  # 1.041198e+13
  
  # model5 <- lm(price_doc ~ poly(full_sq, 4) + as.factor(product_type) + factor(ft_livingtype) + factor(ft_floorType) + as.factor(ft_month) + 
  # as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000) + as.factor(church_count_500)
  # , data=clean.trainDT)
  # 1.037991e+13
  
  ###############
  # PROD TEST FILE
  ###############
  # Take current winner in lm
  # modelPoly <- lm(price_doc ~ poly(full_sq, 4) + as.factor(ft_livingtype) + as.factor(ft_floorType) + 
  #  as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000), data=clean.trainDT,na.action=na.exclude)
  # 0.39219 - !! BEST SUBMISSION
  
  #   modelPoly <- lm(price_doc ~ poly(full_sq, 5) + as.factor(ft_livingtype) + as.factor(ft_floorType)  +
  #  as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000), data=clean.trainDT,na.action=na.exclude)
  # 0.39306
  
}  
  

  
################################################
# SVM - This doesn't produce very good results - ended up getting rid of
################################################  
bRunSVM <- 0
if (bRunSVM ==1)
{
  
  # Expirementing with a singular vector machine
  library(e1071)
  
  # RESULTS
  # lm1 <- lm(price_doc ~ full_sq + as.factor(ft_livingtype) + as.factor(ft_floorType) + 
  #            as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000), data=clean.trainDT,na.action=na.exclude)
  # 0.41398
  
  modelSVM <- svm(price_doc ~ full_sq + as.factor(ft_livingtype) + as.factor(ft_floorType) +
                    as.factor(university_top_20_raion) +  as.factor(sport_count_500) + as.factor(mosque_count_5000), data=clean.trainDT,na.action=na.exclude)
  predSVM <- as.data.frame(predict(modelSVM, clean.testDT))
  
  mean((clean.testDT$price_doc - predict(modelSVM, clean.testDT)) ^ 2) 
  predict()
  
  setnames(predSVM, c("fit"))
  points(clean.trainDT$price_doc, predSVM$fit, col="red", pch=4)
  
  summary(modelSVM)
  
  # With Sub-Areas - This returns 6 negative numbers
  
  nrow(predSVM)
  final.testDT <- cbind(clean.testDT, predSVM)
  final.testDT <- final.testDT[,c("id","fit")]
  final.testDT <- setNames(final.testDT, c("id","price_doc"))
  final.testDT <- final.testDT[order(id),]
  write.csv(final.testDT,file="submission.csv",row.names = F)
  
  summary(final.testDT)
  final.testDT[final.testDT$price_doc < 0]
}  
  
