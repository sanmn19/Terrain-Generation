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
)

REM Iterates throw all text files on %ComplexFolder% and its subfolders.
REM And Populate the array with existent files in this folder and its subfolders
echo     Please wait a while ... We populate the array with complex filesNames ...
SetLocal EnableDelayedexpansion

@FOR /f "delims=" %%f IN ('dir /b /s "%ComplexFolder%\*.png"') DO (
    set /a "idy+=1"
    set "ComplexName[!idy!]=%%~nxf"
    set "ComplexPath[!idy!]=%%~dpFf"
)

rem Display array elements
for /L %%i in (1,1,%idx%) do (
    for /L %%j in (1,1,%idy%) do (    
        echo [%%i] "!FileName[%%i]!"
        ( 
            echo( [%%i] "!FileName[%%i]!"
            echo Path : "!FilePath[%%i]!"
            echo "!ComplexName[%%j]!"
            echo Complex Path : "!ComplexPath[%%j]!"
            echo ************************************
            "D:\Purdue\Research\CGT 521 Fall 2022\Debug\Final Project Terrain Overhang.exe" "!FilePath[%%i]!" "!ComplexPath[%%j]!" "D:\Purdue\Research\CGT 521 Fall 2022\Terrain-Generation\Data Set\Output Images"
        )>> "%LogFile%"
    )
)
ECHO(
ECHO Total text files(s) : !idx!
ECHO Total text files(s) : !idy!
TimeOut /T 10 /nobreak>nul
Start "" "%LogFile%"