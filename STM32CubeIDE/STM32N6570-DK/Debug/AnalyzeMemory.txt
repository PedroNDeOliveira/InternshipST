param (
    [Parameter(Mandatory=$true)]
    [string]$path
)

# Total RAM size in bytes (2047 KB)
$totalRamBytes = 2047 * 1024

# Run size command, filter sections, exclude .ARM.attributes, sum sizes
$usedBytes = arm-none-eabi-size -A $path |
    Select-String -Pattern '^\.(isr_vector|text|gnu\.sgstubs|rodata|version|ARM\.extab|ARM|preinit_array|init_array|fini_array|data|bss|_user_heap_stack)\b' |
    Where-Object { $_ -notmatch 'ARM\.attributes' } |
    ForEach-Object {
        ($_ -split '\s+')[1] -as [int]
    } | Measure-Object -Sum | Select-Object -ExpandProperty Sum

# Calculate remaining RAM in MB with two decimals
$remainingMB = [math]::Round(($totalRamBytes - $usedBytes), 2)

# Output results
Write-Output "File: $ElfFilePath"
Write-Output "Total used bytes in selected sections: $usedBytes"
Write-Output "Remaining RAM: $remainingMB MB"