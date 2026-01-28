/******************************************************************************/
/* Code to replicate and update Romer and Romer (2004) shocks                 */
/*                                                                            */
/* By: Miguel Acosta (modified by Paul Bousquet )                                                         */
/******************************************************************************/

/******************************************************************************/ 
/* Preliminaries                                                              */ 
/******************************************************************************/ 
/* Update Fed-Funds target from FRED again? */ 

/******************************************************************************/
/* Read in Romer & Romer replication material                                 */
/******************************************************************************/
import excel using inputs/RomerandRomerDataAppendix.xls, /*
  */   first clear sheet("DATA BY MEETING")

/* Clean up dates */ 
tostring MTGDATE, replace
replace MTGDATE = "0" + MTGDATE if strlen(MTGDATE)==5
gen fomc = date(MTGDATE,"MD19Y")
replace fomc = mdy(2,11,1987) if fomc == mdy(2,12,1987)

/* Convert to numeric */ 
foreach vv of varlist RESID* GR* IG* {
    destring `vv', replace force 
}


/* Save for later */
tempfile RR
save `RR', replace 

/******************************************************************************/
/* Load Philadelphia Fed Greenbook dataset                                    */
/******************************************************************************/
/* Created in getGBdates.py*/ 
import delimited using intermediates/GBFOMCmapping.csv, /*
  */   stringcols(_all) clear case(preserve)

gen fomc   = date(FOMCdate,"YMD")

/* Merge on each sheet */ 
foreach sheet in gRGDP gPGDP UNEMP {
    preserve 
    import excel intermediates/gbweb_row_format.xlsx, clear first sheet(`sheet')
    cap tostring GBdate, replace 
    tempfile temp
    save `temp', replace
    restore
    merge 1:1 GBdate using `temp', keep(match master) nogen 
}

gen gb = date(GBdate,"YMD")

/* Will need this for determining forecast horizon */ 
gen gbYQ   = yq(year(gb),quarter(gb))

sort fomc

    foreach vv in gRGDP gPGDP UNEMP {
        /* back-cast */ 
       quietly gen     D`vv'B1 = `vv'B1 - `vv'B1[_n-1] /*
        */      if gbYQ == gbYQ[_n-1]
        quietly replace D`vv'B1 = `vv'B1 - `vv'F0[_n-1] /*
        */      if gbYQ >  gbYQ[_n-1]

        /* forecast */ 
        forvalues hh=0/3 {
            local hh1 = `hh' + 1
           quietly gen      D`vv'F`hh' = /*
            */       `vv'F`hh' - `vv'F`hh'[_n-1]  /* 
            */       if gbYQ == gbYQ[_n-1]
			if (`hh'<9) {
				     quietly replace  D`vv'F`hh' = /*
            */       `vv'F`hh' - `vv'F`hh1'[_n-1] /*
            */       if gbYQ >  gbYQ[_n-1]
			}
        }
    }
    
gen daten = mdy(month(fomc), 1, year(fomc))
format daten %td

drop if DATE < 1972 | DATE==.

tempfile gbs
save `gbs', replace

import delimited "https://raw.githubusercontent.com/paulbousquet/GBMPSurprise/main/jk_source_old.csv", clear

gen mr_fomc = date(fomc_latest, "MDY")
format mr_fomc %tdDDmonYY

gen daten = mdy(month(mr_fomc), 1, year(mr_fomc))
format daten %td

gen raw_date = date(date, "MDY")
format mr_fomc %tdDDmonYY

gen raw_daten = mdy(month(raw_date), 1, year(raw_date))
format raw_daten %td

merge m:1 daten using `gbs', nogenerate

* When an unscheduled meeting takes place in the next quarter, scroll forecasts

gen mismatch = (quarter(mr_fomc) != quarter(raw_daten) | year(mr_fomc) != year(raw_daten))

local prefixes "gRGDP DgRGDP gPGDP DgPGDP DUNEMP"

* Loop through each prefix and perform rollover
quietly foreach prefix of local prefixes {
    * Roll B1 <- F0
    replace `prefix'B1 = `prefix'F0 if mismatch == 1
    
    * Roll forward F0, F1, F2 (F0<-F1, F1<-F2, F2<-F3)
    forvalues i = 0/2 {
        local j = `i' + 1
        replace `prefix'F`i' = `prefix'F`j' if mismatch == 1
    }
}

* "edge" cases 
replace gRGDPF3 = gRGDPF4 if mismatch == 1
replace gPGDPF3 = gPGDPF4 if mismatch == 1
replace UNEMPF0 = UNEMPF1 if mismatch == 1

* Keeping with convention to set revision variables to 0 for unscheduled 

local prefixes "DgRGDP DgPGDP DUNEMP"
* Loop through each prefix and perform rollover
quietly foreach prefix of local prefixes {
    * Get list of variables that start with this prefix
    ds `prefix'*
    local varlist `r(varlist)'
    
    * Loop through each variable in the list
    foreach var of local varlist {
        replace `var' = 0 if unscheduled == 1
    }
}

 gen monthly = mofd(raw_daten)
format monthly %tm

gen nzlb1 = monthly < m(2008m11) | monthly > m(2015m5)
gen nzlb2 = monthly < m(2020m3) | monthly > m(2022m8)
gen nzlb = nzlb1 & nzlb2 

drop if unscheduled | !nzlb | monthly < m(1993m1)

sort raw_date
gen t = _n
 
tsset t

export delimited using "prep.csv", replace
