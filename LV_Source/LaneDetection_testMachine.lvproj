<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="21008000">
	<Item Name="My Computer" Type="My Computer">
		<Property Name="NI.SortType" Type="Int">3</Property>
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="FrameWork" Type="Folder">
			<Item Name="Read AVI File.vi" Type="VI" URL="../SubVis/Read AVI File.vi"/>
		</Item>
		<Item Name="Hough_transform" Type="Folder">
			<Item Name="Algorith_source" Type="Folder">
				<Item Name="DrivingLaneDetection-master" Type="Folder">
					<Item Name="test_images" Type="Folder">
						<Item Name="augmented" Type="Folder">
							<Item Name="line-segments-example_augmented.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/augmented/line-segments-example_augmented.jpg"/>
							<Item Name="solidWhiteCurve_augmented.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/augmented/solidWhiteCurve_augmented.jpg"/>
							<Item Name="solidWhiteRight_augmented.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/augmented/solidWhiteRight_augmented.jpg"/>
							<Item Name="solidYellowCurve2_augmented.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/augmented/solidYellowCurve2_augmented.jpg"/>
							<Item Name="solidYellowCurve_augmented.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/augmented/solidYellowCurve_augmented.jpg"/>
							<Item Name="solidYellowLeft_augmented.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/augmented/solidYellowLeft_augmented.jpg"/>
							<Item Name="whiteCarLaneSwitch_augmented.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/augmented/whiteCarLaneSwitch_augmented.jpg"/>
						</Item>
						<Item Name="source" Type="Folder">
							<Item Name="canny_example.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/source/canny_example.jpg"/>
							<Item Name="hough.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/source/hough.jpg"/>
							<Item Name="hough_line.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/source/hough_line.jpg"/>
							<Item Name="hough_polar.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/source/hough_polar.jpg"/>
							<Item Name="hough_polar_1.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/source/hough_polar_1.jpg"/>
							<Item Name="region_mask.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/source/region_mask.jpg"/>
						</Item>
						<Item Name="line-segments-example.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/line-segments-example.jpg"/>
						<Item Name="solidWhiteCurve.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/solidWhiteCurve.jpg"/>
						<Item Name="solidWhiteRight.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/solidWhiteRight.jpg"/>
						<Item Name="solidYellowCurve.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/solidYellowCurve.jpg"/>
						<Item Name="solidYellowCurve2.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/solidYellowCurve2.jpg"/>
						<Item Name="solidYellowLeft.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/solidYellowLeft.jpg"/>
						<Item Name="whiteCarLaneSwitch.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/test_images/whiteCarLaneSwitch.jpg"/>
					</Item>
					<Item Name="test_videos" Type="Folder">
						<Item Name="augmented" Type="Folder">
							<Item Name="extra.mp4" Type="Document" URL="../../../DrivingLaneDetection-master/test_videos/augmented/extra.mp4"/>
							<Item Name="white.mp4" Type="Document" URL="../../../DrivingLaneDetection-master/test_videos/augmented/white.mp4"/>
							<Item Name="yellow.mp4" Type="Document" URL="../../../DrivingLaneDetection-master/test_videos/augmented/yellow.mp4"/>
						</Item>
						<Item Name="challenge.mp4" Type="Document" URL="../../../DrivingLaneDetection-master/test_videos/challenge.mp4"/>
						<Item Name="extra.mp4" Type="Document" URL="../../../DrivingLaneDetection-master/test_videos/extra.mp4"/>
						<Item Name="raw-lines-example.mp4" Type="Document" URL="../../../DrivingLaneDetection-master/test_videos/raw-lines-example.mp4"/>
						<Item Name="solidWhiteRight.mp4" Type="Document" URL="../../../DrivingLaneDetection-master/test_videos/solidWhiteRight.mp4"/>
						<Item Name="solidYellowLeft.mp4" Type="Document" URL="../../../DrivingLaneDetection-master/test_videos/solidYellowLeft.mp4"/>
					</Item>
					<Item Name="_config.yml" Type="Document" URL="../../../DrivingLaneDetection-master/_config.yml"/>
					<Item Name="License.txt" Type="Document" URL="../../../DrivingLaneDetection-master/License.txt"/>
					<Item Name="P1.ipynb" Type="Document" URL="../../../DrivingLaneDetection-master/P1.ipynb"/>
					<Item Name="P1.py" Type="Document" URL="../../../DrivingLaneDetection-master/P1.py"/>
					<Item Name="P2.py" Type="Document" URL="../../../DrivingLaneDetection-master/P2.py"/>
					<Item Name="README.md" Type="Document" URL="../../../DrivingLaneDetection-master/README.md"/>
					<Item Name="solidWhiteCurve.jpg" Type="Document" URL="../../../DrivingLaneDetection-master/solidWhiteCurve.jpg"/>
					<Item Name="Test_image.py" Type="Document" URL="../../../DrivingLaneDetection-master/Test_image.py"/>
				</Item>
			</Item>
			<Item Name="Imaq_to_pythonArray.vi" Type="VI" URL="../python acessors/Imaq_to_pythonArray.vi"/>
		</Item>
		<Item Name="Framework_lib.lvlib" Type="Library" URL="../SubVis/QMH/Framework_lib.lvlib"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="BuildHelpPath.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/BuildHelpPath.vi"/>
				<Item Name="Check Special Tags.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Check Special Tags.vi"/>
				<Item Name="Clear Errors.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Clear Errors.vi"/>
				<Item Name="Color to RGB.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/colorconv.llb/Color to RGB.vi"/>
				<Item Name="Convert property node font to graphics font.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Convert property node font to graphics font.vi"/>
				<Item Name="Details Display Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Details Display Dialog.vi"/>
				<Item Name="DialogType.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/DialogType.ctl"/>
				<Item Name="DialogTypeEnum.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/DialogTypeEnum.ctl"/>
				<Item Name="Error Code Database.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Code Database.vi"/>
				<Item Name="ErrWarn.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/ErrWarn.ctl"/>
				<Item Name="eventvkey.ctl" Type="VI" URL="/&lt;vilib&gt;/event_ctls.llb/eventvkey.ctl"/>
				<Item Name="Find Tag.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Find Tag.vi"/>
				<Item Name="Format Message String.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Format Message String.vi"/>
				<Item Name="General Error Handler Core CORE.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/General Error Handler Core CORE.vi"/>
				<Item Name="General Error Handler.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/General Error Handler.vi"/>
				<Item Name="Get String Text Bounds.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Get String Text Bounds.vi"/>
				<Item Name="Get Text Rect.vi" Type="VI" URL="/&lt;vilib&gt;/picture/picture.llb/Get Text Rect.vi"/>
				<Item Name="GetHelpDir.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/GetHelpDir.vi"/>
				<Item Name="GetRTHostConnectedProp.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/GetRTHostConnectedProp.vi"/>
				<Item Name="Image Type" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/Image Type"/>
				<Item Name="IMAQ ArrayToImage" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ArrayToImage"/>
				<Item Name="IMAQ AVI2 Close" Type="VI" URL="/&lt;vilib&gt;/vision/Avi.llb/IMAQ AVI2 Close"/>
				<Item Name="IMAQ AVI2 Codec Path.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Avi.llb/IMAQ AVI2 Codec Path.ctl"/>
				<Item Name="IMAQ AVI2 Get Info" Type="VI" URL="/&lt;vilib&gt;/vision/Avi.llb/IMAQ AVI2 Get Info"/>
				<Item Name="IMAQ AVI2 Open" Type="VI" URL="/&lt;vilib&gt;/vision/Avi.llb/IMAQ AVI2 Open"/>
				<Item Name="IMAQ AVI2 Read Frame" Type="VI" URL="/&lt;vilib&gt;/vision/Avi.llb/IMAQ AVI2 Read Frame"/>
				<Item Name="IMAQ AVI2 Refnum.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Avi.llb/IMAQ AVI2 Refnum.ctl"/>
				<Item Name="IMAQ Create" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Create"/>
				<Item Name="IMAQ Dispose" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Dispose"/>
				<Item Name="IMAQ GetImageSize" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ GetImageSize"/>
				<Item Name="IMAQ Image.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/IMAQ Image.ctl"/>
				<Item Name="IMAQ ImageToArray" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ImageToArray"/>
				<Item Name="IMAQ Overlay Line" Type="VI" URL="/&lt;vilib&gt;/vision/Overlay.llb/IMAQ Overlay Line"/>
				<Item Name="IMAQ Overlay Multiple Lines 2" Type="VI" URL="/&lt;vilib&gt;/vision/Overlay.llb/IMAQ Overlay Multiple Lines 2"/>
				<Item Name="IMAQ Overlay Oval" Type="VI" URL="/&lt;vilib&gt;/vision/Overlay.llb/IMAQ Overlay Oval"/>
				<Item Name="IMAQ ReadFile 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ ReadFile 2"/>
				<Item Name="IMAQ SetImageSize" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ SetImageSize"/>
				<Item Name="Longest Line Length in Pixels.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Longest Line Length in Pixels.vi"/>
				<Item Name="LVBoundsTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVBoundsTypeDef.ctl"/>
				<Item Name="LVRectTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVRectTypeDef.ctl"/>
				<Item Name="NI_Vision_Development_Module.lvlib" Type="Library" URL="/&lt;vilib&gt;/vision/NI_Vision_Development_Module.lvlib"/>
				<Item Name="Not Found Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Not Found Dialog.vi"/>
				<Item Name="Search and Replace Pattern.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Search and Replace Pattern.vi"/>
				<Item Name="Set Bold Text.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set Bold Text.vi"/>
				<Item Name="Set String Value.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set String Value.vi"/>
				<Item Name="Simple Error Handler.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Simple Error Handler.vi"/>
				<Item Name="TagReturnType.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/TagReturnType.ctl"/>
				<Item Name="Three Button Dialog CORE.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog CORE.vi"/>
				<Item Name="Three Button Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog.vi"/>
				<Item Name="Trim Whitespace.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Trim Whitespace.vi"/>
				<Item Name="whitespace.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/whitespace.ctl"/>
				<Item Name="Get Current LV Bitness.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/Get Current LV Bitness.vi"/>
				<Item Name="System Exec.vi" Type="VI" URL="/&lt;vilib&gt;/Platform/system.llb/System Exec.vi"/>
				<Item Name="Error Cluster From Error Code.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Cluster From Error Code.vi"/>
				<Item Name="IMAQ Copy" Type="VI" URL="/&lt;vilib&gt;/vision/Management.llb/IMAQ Copy"/>
				<Item Name="Image Unit" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/Image Unit"/>
				<Item Name="IMAQ GetImageInfo" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ GetImageInfo"/>
				<Item Name="Edge Polarity.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Measure.llb/Edge Polarity.ctl"/>
				<Item Name="Edge Options.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Measure.llb/Edge Options.ctl"/>
				<Item Name="Edge New.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Measure.llb/Edge New.ctl"/>
				<Item Name="ROI Descriptor" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/ROI Descriptor"/>
				<Item Name="Stall Data Flow.vim" Type="VI" URL="/&lt;vilib&gt;/Utility/Stall Data Flow.vim"/>
				<Item Name="IMAQ Write TIFF File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write TIFF File 2"/>
				<Item Name="IMAQ Write PNG File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write PNG File 2"/>
				<Item Name="IMAQ Write JPEG2000 File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write JPEG2000 File 2"/>
				<Item Name="IMAQ Write JPEG File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write JPEG File 2"/>
				<Item Name="IMAQ Write Image And Vision Info File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write Image And Vision Info File 2"/>
				<Item Name="IMAQ Write BMP File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write BMP File 2"/>
				<Item Name="IMAQ Write File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write File 2"/>
			</Item>
			<Item Name="nivision.dll" Type="Document" URL="nivision.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="nivissvc.dll" Type="Document" URL="nivissvc.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="Main_geometricDetection.vi" Type="VI" URL="../SubVis/Main_geometricDetection.vi"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
