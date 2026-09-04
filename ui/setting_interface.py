from PySide6 import QtWidgets
from qfluentwidgets import (FluentWindow, PushButton, Slider, ProgressBar, PlainTextEdit,
                          setTheme, Theme, FluentIcon, CardWidget, SettingCardGroup,
                          ComboBoxSettingCard, SwitchSettingCard, RangeSettingCard,
                          PushSettingCard, PrimaryPushSettingCard, OptionsSettingCard,
                          FolderListSettingCard, HyperlinkCard, ColorSettingCard, 
                          CustomColorSettingCard)
from backend.config import config, tr, HARDWARD_ACCELERATION_OPTION
from backend.tools.constant import InpaintMode
from backend.tools.hardware_accelerator import HardwareAccelerator

class SettingInterface(QtWidgets.QVBoxLayout):

    def __init__(self, parent):
        super().__init__()
        self.setContentsMargins(16, 16, 16, 16)
        
        # 处理模式设置
        self.processing_profile_combo = ComboBoxSettingCard(
            configItem=config.processingProfile,
            icon=FluentIcon.GLOBE,
            title=tr["SubtitleExtractorGUI"]["ProcessingProfile"],
            content=tr["SubtitleExtractorGUI"]["ProcessingProfileDesc"],
            parent=parent,
            texts=[tr["ModelProfile"]["Basic"], tr["ModelProfile"]["Enhanced"],
                   tr["ModelProfile"]["SttnFast"]],
        )
        self.processing_profile_combo.setToolTip(tr["SubtitleExtractorGUI"]["ProcessingProfileDesc"])
        self.addWidget(self.processing_profile_combo)
        self.processing_profile_combo.comboBox.currentIndexChanged.connect(self._sync_processing_mode)

        # 是否启用硬件加速
        self.hardware_acceleration = SwitchSettingCard(
            configItem=config.hardwareAcceleration,
            icon=FluentIcon.SPEED_HIGH, 
            title=tr["Setting"]["HardwareAcceleration"],
            content=tr["Setting"]["HardwareAccelerationDesc"],
            parent=parent
        )
        self.addWidget(self.hardware_acceleration)
        hardware_available = (HARDWARD_ACCELERATION_OPTION and
                              HardwareAccelerator.instance().has_accelerator())
        if not hardware_available:
            self.hardware_acceleration.switchButton.setChecked(False)
            self.hardware_acceleration.switchButton.setEnabled(False)
            self.hardware_acceleration.setContent(tr["Setting"]["HardwareAccelerationNO"])
        self._sync_processing_mode()
        # 添加一些空间
        self.addStretch(1)
    
    def set_inpaint_mode_enabled(self, enabled):
        """启用或禁用 inpaint 模式下拉框"""
        self.processing_profile_combo.comboBox.setEnabled(enabled)

    def _sync_processing_mode(self, index=None):
        mode_map = {
            'basic': InpaintMode.STTN_DET,
            'enhanced': InpaintMode.PROPAINTER,
            'sttn_fast': InpaintMode.STTN_AUTO,
        }
        mode = mode_map.get(config.processingProfile.value, InpaintMode.STTN_DET)
        config.set(config.inpaintMode, mode)

    def reset_setting(self):
        """重置所有设置为默认值"""
        # 这里需要实现重置逻辑
        pass
