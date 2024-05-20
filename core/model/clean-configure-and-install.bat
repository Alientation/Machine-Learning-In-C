@echo off

REM This script cleans the build and install directories and then configures 
rmdir "./build" /s /q
rmdir "./install" /s /q


REM Configure and build the application
call configure-and-install.bat