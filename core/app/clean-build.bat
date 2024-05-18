@echo off

REM This script cleans the build and install directories and then configures and builds the core application and its dependencies
rmdir "../util/build" /s /q
rmdir "../util/install" /s /q

rmdir "build" /s /q


REM Configure and build the application
call build.bat