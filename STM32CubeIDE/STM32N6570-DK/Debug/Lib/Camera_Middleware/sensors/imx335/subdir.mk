################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
C:/Users/nogueirp/CES_Demo/Lib/Camera_Middleware/sensors/imx335/imx335.c \
C:/Users/nogueirp/CES_Demo/Lib/Camera_Middleware/sensors/imx335/imx335_reg.c 

OBJS += \
./Lib/Camera_Middleware/sensors/imx335/imx335.o \
./Lib/Camera_Middleware/sensors/imx335/imx335_reg.o 

C_DEPS += \
./Lib/Camera_Middleware/sensors/imx335/imx335.d \
./Lib/Camera_Middleware/sensors/imx335/imx335_reg.d 


# Each subdirectory must supply rules for building sources it contributes
Lib/Camera_Middleware/sensors/imx335/imx335.o: C:/Users/nogueirp/CES_Demo/Lib/Camera_Middleware/sensors/imx335/imx335.c Lib/Camera_Middleware/sensors/imx335/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DSTM32N657xx -DUSE_FULL_ASSERT -DUSE_FULL_LL_DRIVER -DVECT_TAB_SRAM -DUSE_IMX335_SENSOR -DUSE_VD66GY_SENSOR -DUSE_VD55G1_SENSOR -DSTM32N6570_DK_REV -DTX_MAX_PARALLEL_NETWORKS=1 -DLL_ATON_PLATFORM=LL_ATON_PLAT_STM32N6 -DLL_ATON_OSAL=LL_ATON_OSAL_FREERTOS -DLL_ATON_RT_MODE=LL_ATON_RT_ASYNC -DLL_ATON_SW_FALLBACK -DLL_ATON_DBG_BUFFER_INFO_EXCLUDED=1 -DAPP_HAS_PARALLEL_NETWORKS=0 -DFEAT_FREERTOS -DSCR_LIB_USE_LTDC -DTRACKER_MODULE -c -I../../../Inc -I../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc -I../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc/Legacy -I../../../STM32Cube_FW_N6/Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../STM32Cube_FW_N6/Drivers/CMSIS/Include -I../../../STM32Cube_FW_N6/Drivers/CMSIS/DSP/Include -I../../../STM32Cube_FW_N6/Drivers/BSP/Components/Common -I../../../STM32Cube_FW_N6/Drivers/BSP/STM32N6570-DK -I../../../STM32Cube_FW_N6/Utilities/lcd -I../../../Lib/AI_Runtime/Inc -I../../../Lib/AI_Runtime/Npu/ll_aton -I../../../Lib/AI_Runtime/Npu/Devices/STM32N6XX -I../../../Lib/lib_vision_models_pp/lib_vision_models_pp/Inc -I../../../Lib/ai-postprocessing-wrapper -I../../../Lib/Camera_Middleware -I../../../Lib/Camera_Middleware/sensors -I../../../Lib/Camera_Middleware/sensors/imx335 -I../../../Lib/Camera_Middleware/sensors/vd55g1 -I../../../Lib/Camera_Middleware/sensors/vd6g -I../../../Lib/Camera_Middleware/ISP_Library/isp/Inc -I../../../Lib/Camera_Middleware/ISP_Library/evision/Inc -I../../../Lib/FreeRTOS/Source/include -I../../../Lib/FreeRTOS/Source/portable/GCC/ARM_CM55_NTZ/non_secure -I../../../Lib/ipl/Inc -I../../../Lib/screenl/Inc -I../../../Lib/tracker -I../../../Lib/NemaGFX/include -I../../../Src -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Lib/Camera_Middleware/sensors/imx335/imx335_reg.o: C:/Users/nogueirp/CES_Demo/Lib/Camera_Middleware/sensors/imx335/imx335_reg.c Lib/Camera_Middleware/sensors/imx335/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DSTM32N657xx -DUSE_FULL_ASSERT -DUSE_FULL_LL_DRIVER -DVECT_TAB_SRAM -DUSE_IMX335_SENSOR -DUSE_VD66GY_SENSOR -DUSE_VD55G1_SENSOR -DSTM32N6570_DK_REV -DTX_MAX_PARALLEL_NETWORKS=1 -DLL_ATON_PLATFORM=LL_ATON_PLAT_STM32N6 -DLL_ATON_OSAL=LL_ATON_OSAL_FREERTOS -DLL_ATON_RT_MODE=LL_ATON_RT_ASYNC -DLL_ATON_SW_FALLBACK -DLL_ATON_DBG_BUFFER_INFO_EXCLUDED=1 -DAPP_HAS_PARALLEL_NETWORKS=0 -DFEAT_FREERTOS -DSCR_LIB_USE_LTDC -DTRACKER_MODULE -c -I../../../Inc -I../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc -I../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc/Legacy -I../../../STM32Cube_FW_N6/Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../STM32Cube_FW_N6/Drivers/CMSIS/Include -I../../../STM32Cube_FW_N6/Drivers/CMSIS/DSP/Include -I../../../STM32Cube_FW_N6/Drivers/BSP/Components/Common -I../../../STM32Cube_FW_N6/Drivers/BSP/STM32N6570-DK -I../../../STM32Cube_FW_N6/Utilities/lcd -I../../../Lib/AI_Runtime/Inc -I../../../Lib/AI_Runtime/Npu/ll_aton -I../../../Lib/AI_Runtime/Npu/Devices/STM32N6XX -I../../../Lib/lib_vision_models_pp/lib_vision_models_pp/Inc -I../../../Lib/ai-postprocessing-wrapper -I../../../Lib/Camera_Middleware -I../../../Lib/Camera_Middleware/sensors -I../../../Lib/Camera_Middleware/sensors/imx335 -I../../../Lib/Camera_Middleware/sensors/vd55g1 -I../../../Lib/Camera_Middleware/sensors/vd6g -I../../../Lib/Camera_Middleware/ISP_Library/isp/Inc -I../../../Lib/Camera_Middleware/ISP_Library/evision/Inc -I../../../Lib/FreeRTOS/Source/include -I../../../Lib/FreeRTOS/Source/portable/GCC/ARM_CM55_NTZ/non_secure -I../../../Lib/ipl/Inc -I../../../Lib/screenl/Inc -I../../../Lib/tracker -I../../../Lib/NemaGFX/include -I../../../Src -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Lib-2f-Camera_Middleware-2f-sensors-2f-imx335

clean-Lib-2f-Camera_Middleware-2f-sensors-2f-imx335:
	-$(RM) ./Lib/Camera_Middleware/sensors/imx335/imx335.cyclo ./Lib/Camera_Middleware/sensors/imx335/imx335.d ./Lib/Camera_Middleware/sensors/imx335/imx335.o ./Lib/Camera_Middleware/sensors/imx335/imx335.su ./Lib/Camera_Middleware/sensors/imx335/imx335_reg.cyclo ./Lib/Camera_Middleware/sensors/imx335/imx335_reg.d ./Lib/Camera_Middleware/sensors/imx335/imx335_reg.o ./Lib/Camera_Middleware/sensors/imx335/imx335_reg.su

.PHONY: clean-Lib-2f-Camera_Middleware-2f-sensors-2f-imx335

