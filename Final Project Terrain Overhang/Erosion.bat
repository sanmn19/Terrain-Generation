@echo off
Title Populate Array with filenames and show them
set "MasterFolder=D:\Purdue\Research\CGT 521 Fall 2022\Terrain-Generation\Data Set\Heightmaps"
set "ComplexFolder=D:\Purdue\Research\CGT 521 Fall 2022\Terrain-Generation\Data Set\Complex Images"
Set LogFile=%~dpn0.txt
If exist "%LogFile%" Del "%LogFile%"

REM Iterates throw all text files on %MasterFolder% and its subfolders.
REM And Populate the array with existent files in this folder and its subfolders
echo     Please wait a while ... We populate the array with filesNames ...
SetLocal EnableDelayedexpansion

@FOR /f "delims=" %%f IN ('dir /b /s "%MasterFolder%\*.png"') DO (
    set /a "idx+=1"
    set "FileName[!idx!]=%%~nxf"
    set "FilePath[!idx!]=%%~dpFf"
    set "FileNameNE[!idx!]=%%~nf"
)

REM Iterates throw all text files on %ComplexFolder% and its subfolders.
REM And Populate the array with existent files in this folder and its subfolders
echo     Please wait a while ... We populate the array with complex filesNames ...
SetLocal EnableDelayedexpansion

@FOR /f "delims=" %%f IN ('dir /b /s "%ComplexFolder%\*.png"') DO (
    set /a "idy+=1"
    set "ComplexName[!idy!]=%%~nxf"
    set "ComplexPath[!idy!]=%%~dpFf"
    set "ComplexNameNE[!idy!]=%%~nf"
)

setlocal EnableDelayedExpansion
for /L %%x in (1,1,%idx%) do (
    set /A randox=!random! %% !idy!
    set /A randoy=!random! %% !idx!
    set "randoxArray[%%x]=!randox!"
    set "randoyArray[%%x]=!randoy!"
)

setlocal EnableDelayedExpansion
rem Display array elements
for /L %%j in (1,1,%idy%) do (
    for /L %%i in (1,1,%idx%) do (
        set /A randotempx=!randoxArray[%%i]!
        set /A randoy=!random! %% !idx!
    	set currentHeightMap=!FileName[!randotempx!]!
    	set currentHeightMapPath=!FilePath[!randoxArray[%%i]!]!
    	set currentHeightMapNE=!FileNameNE[!randoxArray[%%i]!]!
    	set currentComplexMap=!ComplexName[!randoyArray[%%j]!]!
    	set currentComplexMapPath=!ComplexPath[!randoyArray[%%j]!]!
    	set currentComplexMapNE=!ComplexNameNE[!randoyArray[%%j]!]!
        
	echo [!randotempx!] "!currentHeightMap!"
        ( 
            echo( [!randox!] "!currentHeightMap!"
            echo Path : "!currentHeightMapPath!"
            echo "!currentHeightMapNE!"
            echo "!currentComplexMap!"
            echo Complex Path : "!currentComplexMapPath!"
            echo "!currentComplexMapNE!"
            echo ************************************
            set "outPath=D:\Purdue\Research\CGT 521 Fall 2022\Terrain-Generation\Data Set\Output Images\!currentHeightMapNE!_!currentComplexMapNE!_Out.png"
            echo "!outPath!"
            If exist "!outPath!" (echo "Already exists") ELSE ("D:\Purdue\Research\CGT 521 Fall 2022\Debug\Final Project Terrain Overhang.exe" "!currentHeightMapPath!" "!currentComplexMapPath!" "D:\Purdue\Research\CGT 521 Fall 2022\Terrain-Generation\Data Set\Output Images")
        )>> "%LogFile%"
    )
)

ECHO(
ECHO Total text files(s) : !idx!
ECHO Total text files(s) : !idy!
TimeOut /T 10 /nobreak>nul
Start "" "%LogFile%"