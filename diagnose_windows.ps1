# Read-only USB diagnostics for SpinQ Gemini Lab on Windows 10/11.
# Run while the instrument is idle. No driver, device command, or admin rights.
param([switch]$Compare)

$ErrorActionPreference = 'Stop'

function Get-Snapshot {
    $devices = @(Get-PnpDevice -PresentOnly | Where-Object {
        $_.InstanceId -match '^(USB|FTDIBUS)\\'
    })
    $ports = @(Get-CimInstance Win32_SerialPort | Select-Object DeviceID, Name)
    $usbNetwork = @(Get-NetAdapter | Where-Object {
        $_.InterfaceDescription -match 'USB|RNDIS|Remote NDIS'
    } | Select-Object Name, InterfaceDescription, Status)
    return [pscustomobject]@{
        Devices = $devices
        Ports = $ports
        UsbNetwork = $usbNetwork
    }
}

function Show-Device($item) {
    $vendorId = if ($item.InstanceId -match 'VID_([0-9A-Fa-f]{4})') { $Matches[1] } else { '?' }
    $productId = if ($item.InstanceId -match 'PID_([0-9A-Fa-f]{4})') { $Matches[1] } else { '?' }
    [pscustomobject]@{
        Name = $item.FriendlyName
        Class = $item.Class
        VID = $vendorId
        PID = $productId
        Status = $item.Status
    }
}

function Show-Snapshot($snapshot) {
    Write-Host "`nUSB zariadenia (bez sériových čísel):"
    $display = @($snapshot.Devices | ForEach-Object { Show-Device $_ })
    if ($display.Count) { $display | Format-Table -AutoSize | Out-Host }
    else { Write-Host '(žiadne)' }

    Write-Host "`nSériové porty:"
    if ($snapshot.Ports.Count) { $snapshot.Ports | Format-Table -AutoSize | Out-Host }
    else { Write-Host '(žiadne)' }

    Write-Host "`nUSB sieťové adaptéry:"
    if ($snapshot.UsbNetwork.Count) { $snapshot.UsbNetwork | Format-Table -AutoSize | Out-Host }
    else { Write-Host '(žiadne)' }
}

try {
    if ($Compare) {
        Write-Host 'Odpoj USB kábel od Windows PC a stlač Enter.'
        [void](Read-Host)
        $before = Get-Snapshot
        Write-Host 'Zapoj USB kábel do Windows PC a stlač Enter.'
        [void](Read-Host)
        Start-Sleep -Seconds 2
        $after = Get-Snapshot
        $beforeIds = @($before.Devices | ForEach-Object { $_.InstanceId })
        $newDevices = @($after.Devices | Where-Object { $beforeIds -notcontains $_.InstanceId })
        Write-Host "`nNové zariadenia po zapojení:"
        if ($newDevices.Count) { $newDevices | ForEach-Object { Show-Device $_ } | Format-Table -AutoSize | Out-Host }
        else { Write-Host '(žiadne nové USB zariadenie)' }
        Show-Snapshot $after
    } else {
        Show-Snapshot (Get-Snapshot)
    }
} catch {
    Write-Error "Diagnostika zlyhala: $($_.Exception.Message)"
    exit 1
}
