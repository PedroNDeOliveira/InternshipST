################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
C:/Users/nogueirp/CES_Demo/Gcc/Src/console.c \
C:/Users/nogueirp/CES_Demo/Gcc/Src/freertos_libc.c \
C:/Users/nogueirp/CES_Demo/Gcc/Src/syscalls.c 

OBJS += \
./Gcc/Src/console.o \
./Gcc/Src/freertos_libc.o \
./Gcc/Src/syscalls.o 

C_DEPS += \
./Gcc/Src/console.d \
./Gcc/Src/freertos_libc.d \
./Gcc/Src/syscalls.d 


# Each subdirectory must supply rules for building sources it contributes
Gcc/Src/console.o: C:/Users/nogueirp/CES_Demo/Gcc/Src/console.c Gcc/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DSTM32N657xx -DUSE_FULL_ASSERT -DUSE_FULL_LL_DRIVER -DVECT_TAB_SRAM -DUSE_IMX335_SENSOR -DUSE_VD66GY_SENSOR -DUSE_VD55G1_SENSOR -DSTM32N6570_DK_REV -DTX_MAX_PARALLEL_NETWORKS=1 -DLL_ATON_PLATFORM=LL_ATON_PLAT_STM32N6 -DLL_ATON_OSAL=LL_ATON_OSAL_FREERTOS -DLL_ATON_RT_MODE=LL_ATON_RT_ASYNC -DLL_ATON_SW_FALLBACK -DLL_ATON_DBG_BUFFER_INFO_EXCLUDED=1 -DAPP_HAS_PARALLEL_NETWORKS=0 -DFEAT_FREERTOS -DSCR_LIB_USE_LTDC -DTRACKER_MODULE -c -I../../../Inc -I../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc -I../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc/Legacy -I../../../STM32Cube_FW_N6/Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../STM32Cube_FW_N6/Drivers/CMSIS/Include -I../../../STM32Cube_FW_N6/Drivers/CMSIS/DSP/Include -I../../../STM32Cube_FW_N6/Drivers/BSP/Components/Common -I../../../STM32Cube_FW_N6/Drivers/BSP/STM32N6570-DK -I../../../STM32Cube_FW_N6/Utilities/lcd -I../../../Lib/AI_Runtime/Inc -I../../../Lib/AI_Runtime/Npu/ll_aton -I../../../Lib/AI_Runtime/Npu/Devices/STM32N6XX -I../../../Lib/lib_vision_models_pp/lib_vision_models_pp/Inc -I../../../Lib/ai-postprocessing-wrapper -I../../../Lib/Camera_Middleware -I../../../Lib/Camera_Middleware/sensors -I../../../Lib/Camera_Middleware/sensors/imx335 -I../../../Lib/Camera_Middleware/sensors/vd55g1 -I../../../Lib/Camera_Middleware/sensors/vd6g -I../../../Lib/Camera_Middleware/ISP_Library/isp/Inc -I../../../Lib/Camera_Middleware/ISP_Library/evision/Inc -I../../../Lib/FreeRTOS/Source/include -I../../../Lib/FreeRTOS/Source/portable/GCC/ARM_CM55_NTZ/non_secure -I../../../Lib/ipl/Inc -I../../../Lib/screenl/Inc -I../../../Lib/tracker -I../../../Lib/NemaGFX/include -I../../../Src -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Gcc/Src/freertos_libc.o: C:/Users/nogueirp/CES_Demo/Gcc/Src/freertos_libc.c Gcc/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DSTM32N657xx -DUSE_FULL_ASSERT -DUSE_FULL_LL_DRIVER -DVECT_TAB_SRAM -DUSE_IMX335_SENSOR -DUSE_VD66GY_SENSOR -DUSE_VD55G1_SENSOR -DSTM32N6570_DK_REV -DTX_MAX_PARALLEL_NETWORKS=1 -DLL_ATON_PLATFORM=LL_ATON_PLAT_STM32N6 -DLL_ATON_OSAL=LL_ATON_OSAL_FREERTOS -DLL_ATON_RT_MODE=LL_ATON_RT_ASYNC -DLL_ATON_SW_FALLBACK -DLL_ATON_DBG_BUFFER_INFO_EXCLUDED=1 -DAPP_HAS_PARALLEL_NETWORKS=0 -DFEAT_FREERTOS -DSCR_LIB_USE_LTDC -DTRACKER_MODULE -c -I../../../Inc -I../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc -I../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc/Legacy -I../../../STM32Cube_FW_N6/Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../STM32Cube_FW_N6/Drivers/CMSIS/Include -I../../../STM32Cube_FW_N6/Drivers/CMSIS/DSP/Include -I../../../STM32Cube_FW_N6/Drivers/BSP/Components/Common -I../../../STM32Cube_FW_N6/Drivers/BSP/STM32N6570-DK -I../../../STM32Cube_FW_N6/Utilities/lcd -I../../../Lib/AI_Runtime/Inc -I../../../Lib/AI_Runtime/Npu/ll_aton -I../../../Lib/AI_Runtime/Npu/Devices/STM32N6XX -I../../../Lib/lib_vision_models_pp/lib_vision_models_pp/Inc -I../../../Lib/ai-postprocessing-wrapper -I../../../Lib/Camera_Middleware -I../../../Lib/Camera_Middleware/sensors -I../../../Lib/Camera_Middleware/sensors/imx335 -I../../../Lib/Camera_Middleware/sensors/vd55g1 -I../../../Lib/Camera_Middleware/sensors/vd6g -I../../../Lib/Camera_Middleware/ISP_Library/isp/Inc -I../../../Lib/Camera_Middleware/ISP_Library/evision/Inc -I../../../Lib/FreeRTOS/Source/include -I../../../Lib/FreeRTOS/Source/portable/GCC/ARM_CM55_NTZ/non_secure -I../../../Lib/ipl/Inc -I../../../Lib/screenl/Inc -I../../../Lib/tracker -I../../../Lib/NemaGFX/include -I../../../Src -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Gcc/Src/syscalls.o: C:/Users/nogueirp/CES_Demo/Gcc/Src/syscalls.c Gcc/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DSTM32N657xx -DUSE_FULL_ASSERT -DUSE_FULL_LL_DRIVER -DVECT_TAB_SRAM -DUSE_IMX335_SENSOR -DUSE_VD66GY_SENSOR -DUSE_VD55G1_SENSOR -DSTM32N6570_DK_REV -DTX_MAX_PARALLEL_NETWORKS=1 -DLL_ATON_PLATFORM=LL_ATON_PLAT_STM32N6 -DLL_ATON_OSAL=LL_ATON_OSAL_FREERTOS -DLL_ATON_RT_MODE=LL_ATON_RT_ASYNC -DLL_ATON_SW_FALLBACK -DLL_ATON_DBG_BUFFER_INFO_EXCLUDED=1 -DAPP_HAS_PARALLEL_NETWORKS=0 -DFEAT_FREERTOS -DSCR_LIB_USE_LTDC -DTRACKER_MODULE -c -I../../../Inc -I../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc -I../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc/Legacy -I../../../STM32Cube_FW_N6/Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../STM32Cube_FW_N6/Drivers/CMSIS/Include -I../../../STM32Cube_FW_N6/Drivers/CMSIS/DSP/Include -I../../../STM32Cube_FW_N6/Drivers/BSP/Components/Common -I../../../STM32Cube_FW_N6/Drivers/BSP/STM32N6570-DK -I../../../STM32Cube_FW_N6/Utilities/lcd -I../../../Lib/AI_Runtime/Inc -I../../../Lib/AI_Runtime/Npu/ll_aton -I../../../Lib/AI_Runtime/Npu/Devices/STM32N6XX -I../../../Lib/lib_vision_models_pp/lib_vision_models_pp/Inc -I../../../Lib/ai-postprocessing-wrapper -I../../../Lib/Camera_Middleware -I../../../Lib/Camera_Middleware/sensors -I../../../Lib/Camera_Middleware/sensors/imx335 -I../../../Lib/Camera_Middleware/sensors/vd55g1 -I../../../Lib/Camera_Middleware/sensors/vd6g -I../../../Lib/Camera_Middleware/ISP_Library/isp/Inc -I../../../Lib/Camera_Middleware/ISP_Library/evision/Inc -I../../../Lib/FreeRTOS/Source/include -I../../../Lib/FreeRTOS/Source/portable/GCC/ARM_CM55_NTZ/non_secure -I../../../Lib/ipl/Inc -I../../../Lib/screenl/Inc -I../../../Lib/tracker -I../../../Lib/NemaGFX/include -I../../../Src -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Gcc-2f-Src

clean-Gcc-2f-Src:
	-$(RM) ./Gcc/Src/console.cyclo ./Gcc/Src/console.d ./Gcc/Src/console.o ./Gcc/Src/console.su ./Gcc/Src/freertos_libc.cyclo ./Gcc/Src/freertos_libc.d ./Gcc/Src/freertos_libc.o ./Gcc/Src/freertos_libc.su ./Gcc/Src/syscalls.cyclo ./Gcc/Src/syscalls.d ./Gcc/Src/syscalls.o ./Gcc/Src/syscalls.su

.PHONY: clean-Gcc-2f-Src

