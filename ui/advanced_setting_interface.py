"""Application settings and support page."""

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import (
    ScrollArea, ExpandLayout, SettingCardGroup, HyperlinkCard,
    PrimaryPushSettingCard, PushSettingCard, SwitchSettingCard,
    FluentIcon, MessageBox, BodyLabel, SubtitleLabel
)

from backend.config import config, tr, VERSION, PROJECT_HOME_URL, PROJECT_ISSUES_URL, PROJECT_RELEASES_URL
from backend.tools.version_service import VersionService
from backend.tools.concurrent import TaskExecutor


class AdvancedSettingInterface(ScrollArea):
    """A compact settings surface for output, updates and support."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.version_manager = VersionService()
        self._build()

    def _build(self):
        self.scrollWidget = QtWidgets.QWidget(self)
        self.expandLayout = ExpandLayout(self.scrollWidget)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        intro_widget = QtWidgets.QWidget(self.scrollWidget)
        intro = QtWidgets.QVBoxLayout(intro_widget)
        intro.setContentsMargins(20, 18, 20, 6)
        title = SubtitleLabel(tr["Setting"]["AdvancedSetting"], self.scrollWidget)
        description = BodyLabel(tr["Setting"]["SettingsIntro"], self.scrollWidget)
        description.setObjectName("settingsIntro")
        intro.addWidget(title)
        intro.addWidget(description)
        self.expandLayout.addWidget(intro_widget)

        self.performance_group = SettingCardGroup(tr["Setting"]["PerformanceOutput"], self.scrollWidget)
        self.save_directory = PushSettingCard(
            text=tr["Setting"]["ChooseDirectory"], icon=FluentIcon.DOWNLOAD,
            title=tr["Setting"]["SaveDirectory"],
            content=tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value,
            parent=self.performance_group)
        self.save_directory.clicked.connect(self.choose_save_directory)
        self.check_update_on_startup = SwitchSettingCard(
            configItem=config.checkUpdateOnStartup, icon=FluentIcon.UPDATE,
            title=tr["Setting"]["CheckUpdateOnStartup"],
            content=tr["Setting"]["CheckUpdateOnStartupDesc"], parent=self.performance_group)
        self.performance_group.addSettingCard(self.save_directory)
        self.performance_group.addSettingCard(self.check_update_on_startup)
        self.expandLayout.addWidget(self.performance_group)

        self.about_group = SettingCardGroup(tr["Setting"]["AboutSetting"], self.scrollWidget)
        self.feedback = PrimaryPushSettingCard(
            text=tr["Setting"]["FeedbackButton"], icon=FluentIcon.MAIL,
            title=tr["Setting"]["FeedbackTitle"], content=tr["Setting"]["FeedbackDesc"], parent=self.about_group)
        self.feedback.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(PROJECT_ISSUES_URL)))
        self.copyright = PrimaryPushSettingCard(
            text=tr["Setting"]["CopyrightButton"], icon=FluentIcon.INFO,
            title=tr["Setting"]["CopyrightTitle"], content=tr["Setting"]["CopyrightDesc"].format(VERSION), parent=self.about_group)
        self.copyright.clicked.connect(self.check_update)
        self.project_link = HyperlinkCard(
            url=PROJECT_HOME_URL, text=PROJECT_HOME_URL, icon=FluentIcon.GITHUB,
            title=tr["Setting"]["ProjectLinkTitle"], content=tr["Setting"]["ProjectLinkDesc"], parent=self.about_group)
        self.about_group.addSettingCard(self.feedback)
        self.about_group.addSettingCard(self.copyright)
        self.about_group.addSettingCard(self.project_link)
        self.expandLayout.addWidget(self.about_group)
        self.expandLayout.setSpacing(14)
        self.expandLayout.setContentsMargins(16, 6, 16, 36)

    def show_message_box(self, title, content, showYesButton=False, yesSlot=None):
        box = MessageBox(title, content, self)
        if not showYesButton:
            box.cancelButton.setText(self.tr("Close"))
            box.yesButton.hide()
            box.buttonLayout.insertStretch(0, 1)
        if box.exec() and yesSlot is not None:
            yesSlot()

    def check_update(self, ignore=False):
        TaskExecutor.runTask(self.version_manager.has_new_version).then(
            lambda success: self.on_version_info_fetched(success, ignore))

    def on_version_info_fetched(self, success, ignore=False):
        if success:
            self.show_message_box(
                tr["Setting"]["UpdatesAvailableTitle"],
                tr["Setting"]["UpdatesAvailableDesc"].format(self.version_manager.lastest_version),
                True, lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(PROJECT_RELEASES_URL)))
        elif not ignore:
            self.show_message_box(tr["Setting"]["NoUpdatesAvailableTitle"], tr["Setting"]["NoUpdatesAvailableDesc"])

    def choose_save_directory(self):
        last_directory = "." if not config.saveDirectory.value else config.saveDirectory.value
        folder = QFileDialog.getExistingDirectory(self, tr["Setting"]["ChooseDirectory"], last_directory)
        if folder:
            config.set(config.saveDirectory, folder)
            self.save_directory.setContent(folder)
