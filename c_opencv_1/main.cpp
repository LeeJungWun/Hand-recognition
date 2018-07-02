#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <iostream>
#include "stdafx.h"
#include "BlobLabeling.h"
#include <stdio.h>

using namespace std;
using namespace cv;

CvPoint pt;
CvPoint fst;

POINT point_i = { 500, 500 }, point_s = { 0, 0 };

typedef struct _WhitePoints
{
	int num;
	int x[100000];
	int y[100000];
};

typedef struct _centerPoints {
	int center_x;
	int center_y;
} centerPoints;




int main()
{
	point_i.x = 0; point_i.y = 0;
	int locationvalue = 1;
	int locationpoint = 0;
	int countall = 0;         //좌표값의 전체적인 출력 카운트
	int count = 0;
	int counta = 0;
	int sum_x, sum_y;
	int countlocation = 0;      //좌표가 유지된는지를 알기위한 카운트
	int prevX = 0;
	int prevY = 0;
	int eye = 0;
	int nose = 0;
	int mouth = 0;
	int blob_count = 0;
	int facecount = 0;
	int handcount = 0;
	int average_x = 0, average_y = 0;
	int mouseup_count = 0;
	int breakcount = 0;				//키보드 이벤트를 윈한 카운트
	int clickcount = 0;				//클릭 이벤트를 추가하기 위한 카운트

	int total_width = GetSystemMetrics(SM_CYSCREEN);	//화면 전체 해상도의 가로길이
	int total_height = GetSystemMetrics(SM_CXSCREEN);	//화면 전체 해상도의 세로길이
	clock_t start, finish;
	double duration = 0.0;

	int view = 0;

	FILE *f;

	char s_output_result[50];
	CvFont font;

	VideoCapture camera(0);


	Mat input(480, 640, CV_8UC3);
	Mat output(480, 640, CV_8UC3);
	Mat img_thresh(480, 640, CV_8UC1);
	Mat morphology(480, 640, CV_8UC1);
	Mat binary(480, 640, CV_8UC1);



	Scalar hsv_min = cvScalar(0, 30, 60, 0);  //네 개의 double형 실수값을 배열 형태로 가지고 있다. 메모리 사용량이 크게 문제되지 않은 경우사용 한다.
	Scalar hsv_max = cvScalar(30, 255, 255, 0);  //min=피부색의 최소값 max=피부색 최대값

	IplImage *Ipl_skin = new IplImage(binary);
	IplImage *trans = &IplImage(input);


	/*
	const char *classifer = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";               //얼굴을 인식하는 xml파일 열기
	const char *classifer1 = "C:/opencv/sources/data/haarcascades/haarcascade_mcs_nose.xml";                  //코를 인식하는 xml파일 열기
	const char *classifer2 = "C:/opencv/sources/data/haarcascades/haarcascade_mcs_mouth.xml";                  //입을 인식하는 xml파일 열기
	const char *classifer3 = "C:/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";            //눈을 인식하는 xml파일 열기

	CvHaarClassifierCascade* cascade_face = 0;      //cacscade_face 값 초기화
	cascade_face = (CvHaarClassifierCascade*)cvLoad(classifer, 0, 0, 0);   //xml파일을 불러온다.

	CvHaarClassifierCascade* cascade_nose = 0;
	cascade_nose = (CvHaarClassifierCascade*)cvLoad(classifer1, 0, 0, 0);

	CvHaarClassifierCascade* cascade_mouth = 0;
	cascade_mouth = (CvHaarClassifierCascade*)cvLoad(classifer2, 0, 0, 0);

	CvHaarClassifierCascade* cascade_eye = 0;
	cascade_eye = (CvHaarClassifierCascade*)cvLoad(classifer3, 0, 0, 0);

	if (!cascade_face || !cascade_nose || !cascade_mouth || !cascade_eye) {   //사용할 xml 파일이 없으면 에러
		std::cerr << "error: cascade error!!" << std::endl;
		return -1;
	}
	*/
	/*
	CvMemStorage* storage_face = 0;
	storage_face = cvCreateMemStorage(0);   //메모리 스토리지를 생성하기 위해 사용한다. 값을 0으로 하면 64KB의 기본 블록 크기가 사용된다.

	CvMemStorage* storage_nose = 0;
	storage_nose = cvCreateMemStorage(0);

	CvMemStorage* storage_mouth = 0;
	storage_mouth = cvCreateMemStorage(0);

	CvMemStorage* storage_eye = 0;
	storage_eye = cvCreateMemStorage(0);

	if (!storage_face || !storage_nose || !storage_mouth || !storage_eye) {   //메모리 스토리지가 없으면 에러
		std::cerr << "error: storage error!!" << std::endl;
		return -2;
	}
	*/


	//----------------------------------------------------------
	printf("%d", total_width);
	printf("%d", total_height);
	for (;;)
	{
		if (breakcount >= 1)
		{
			//facecount = 0;
			breakcount = 0;
			//eye = 0;
			//nose = 0;
			//mouth = 0;
		}
		sum_x = 0, sum_y = 0;

		_WhitePoints cp;
		cp.num = 0;
		cp.x[100000];
		cp.y[100000];

		camera >> input;
		//printf("%d\n", facecount);
		
		//----------------------------------
		//for (; facecount <= 20; facecount++)
		//{
			//cvDestroyWindow("input");
			//cvDestroyWindow("output");
			//cvDestroyWindow("threshold");
			//cvDestroyWindow("morphology");

			//printf("Face Recognition");
			IplImage* image = 0;
			CvCapture* cap = cvCaptureFromCAM(0);      //PC카메라로부터 동영상을 캡쳐하는데 사용하는 함수이다.



			cvGrabFrame(cap);            //동영상 파일로 부터 하나하나의 프레임을 잡는데 사용하는 합수이다.
			image = cvRetrieveFrame(cap);   //cvGrabFrame() 함수로부터 잡은 프레임으로부터 영상을 얻어내는데 사용하는 함수이다.
			cvResizeWindow("cam", 100, 100);

			//CvSeq *faces = 0;

			//---------------------------------------------------------------------------------------------




			//CvPoint pt1 = cvPoint(200, 200); //m_recBlobs; 
			//CvPoint pt2 = cvPoint(pt1.x + 300, pt1.y + 500);
			//CvScalar color1 = cvScalar(255, 0, 0); // B,G,R // 빨간색
			//cvDrawRect(image, pt1, pt2, color1, 2); // 사각형 그리기, 원본, 시작점, 끝점,칼라, 선 두께

													//------------------------------------------------------------------------------------------------

			//faces = cvHaarDetectObjects(image, cascade_face, storage_face, 1.5, 2, 2);

			//(대상 object를 찾는 작업을 하고자 하는 원본 이미지, sample들이 학습된 xml file, 메모리공간, 찾고자하는 대상의 스케일 변화 정도, Object라 판정된 결과가 minNeighbors만큼 겹치는 경우에만 최종적으로 Object로 판단하도록 하는 변수. Default 값은 3이다., 4가지 설정요소가 있다.,  min = cvSize(0,0),  max = cvSize(0,0))
			//CV_HAAR_DO_CANNY_PRUNING: 아무 선이 없는 평평한 부분을 생략한다.
			//CV_HAAR_SCALE_IMAGE: 이미지의 스케일을 변화시키도록 한다.
			//CV_HAAR_FIND_BIGGEST_OBJECT : 가장 큰 Object만 감지한다.
			//CV_HAAR_DO_ROUGH_SEARCH : 크기와 무관하게 첫 번째 Object 후보가 감지되었을 때, 검색을 중지한다.

			/*

			for (int i = 0; i < faces->total; i++) {

				CvRect *r = 0;
				r = (CvRect*)cvGetSeqElem(faces, i);

				//fascs : 찾고자하는 원본 sequence
				//i : 찿고자하는 원소의 index
				//index i에 따라 sequence faces의 원소들의 포인터를 반환한다.
				//만약 faces의 원소가 발견되지 않았다면 0을 반환한다.
				//함수는 음의 숫자를 지원한다. -1은 faces의 마지막 원소를 의미하고 -2는 뒤에서 두번째 원소를 의미한다.

				cvRectangle(image, cvPoint(r->x, r->y), cvPoint(r->x + r->width, r->y + r->height), cvScalar(0, 255, 0), 3, CV_AA, 0);

				//직사각형을 그리기위한 함수이다.
				//(직사각형이 그려질 영상을 나타낸다., 직사각형의 모서리를 나타낸다., 직사각형의 반대편 모서리를 나타낸다., 직사각형의 성의 색상을 나타낸다., 직사각형의 두께를 나타낸다., 선의 형태, 0)

				CvRect rect;
				rect.x = r->x;
				rect.y = r->y;
				rect.width = r->width;
				rect.height = r->height;
				printf("x=%d, y=%d, width=%d, height=%d\n", rect.x, rect.y, rect.width, rect.height);      //얼굴이 인식되는 좌표 출력

				if (rect.x >= 120 && rect.x <= 350 && rect.y >= 240)         // x좌표 120 ~ 350 y좌표 240이상 일때 눈,코,입을 인식하도록한다.
				{
					printf("area x=%d, y=%d, width=%d, height=%d\n", rect.x, rect.y, rect.width, rect.height);      //위 조건 확인을 위한 출력문
					cvSetImageROI(image, rect);
					//image에 rect만큼 영역 설정을 하고 나면 image는 내가 선택한 영역만 신경쓰게 된다.
					IplImage* fimg = cvCreateImage(cvSize(r->width, r->height), image->depth, image->nChannels);
					//영상을 만들기 위해 사용하는 함수이다.
					//cvSize(r->width, r->height) : 영상의 크기를 지정한다. 영상의 크기는 cvSize() 함수와 cvGetSize() 함수를 사용하여 지정한다. 영상의 가로와 세로 길이를 직접 지정하고자 하는 경우는 cvSize() 함수를 사용하여 cvSize(width, height)처럼 지정하고 특정한 영상의 크기와 똑같은 크기의 영상을 만들고자 하는 경우는 cvGetSize() 함수를 사용하여 cvGetSize(src_image)처럼 지정한다.
					//image->depth : 영상 데이터의 깊이(단위:비트)를 지정한다. 1채널 영상인 경우는 픽셀의 깊이가 되고 3채널 영상인 경우 각 채널의 깊이가 된다.
					//픽셀 당 채널 수를 지정한다. 예를 들어, 영상의 크기 640 * 480인 흑백영상을 만들고자 하는 경우 흑백영상은 1채널이므로 다음과 같다.
					cvCopy(image, fimg);
					//새로운 fimg를 생성하고 관심영역이 설정된 image를 fimg로 카피
					cvResetImageROI(image);
					//Image는 원래대로 복원.




					CvSeq *noses = 0;

					noses = cvHaarDetectObjects(fimg, cascade_nose, storage_nose, 2.0, 5, 1);
					for (int j = 0; j < noses->total; j++) {
						CvRect *r1 = 0;
						r1 = (CvRect*)cvGetSeqElem(noses, j);

						cvRectangle(image, cvPoint(r1->x + r->x, r1->y + r->y), cvPoint(r1->x + r->x + r1->width, r1->y + r->y + r1->height), cvScalar(0, 0, 255), 3, CV_AA, 0);
						nose++;      //코가 몇번 인식이 되었는지 카운트
					}//코를 인식하는 부분




					CvSeq *mouths = 0;
					mouths = cvHaarDetectObjects(fimg, cascade_mouth, storage_mouth, 2.0, 5, 1);

					for (int k = 0; k < mouths->total; k++)
					{
						CvRect * r2 = 0;
						r2 = (CvRect*)cvGetSeqElem(mouths, k);
						cvRectangle(image, cvPoint(r2->x + r->x, r2->y + r->y), cvPoint(r2->x + r->x + r2->width, r2->y + r->y + r2->height), cvScalar(255, 0, 0), 3, CV_AA, 0);
						mouth++;      //입이 몇번 인식이 되었는지 카운트
					}//입을 인식하기위한 부분





					CvSeq *eyes = 0;
					eyes = cvHaarDetectObjects(fimg, cascade_eye, storage_eye, 2.0, 5, 1);

					for (int l = 0; l < eyes->total; l++)
					{
						CvRect * r3 = 0;
						r3 = (CvRect*)cvGetSeqElem(eyes, l);
						cvRectangle(image, cvPoint(r3->x + r->x, r3->y + r->y), cvPoint(r3->x + r->x + r3->width, r3->y + r->y + r3->height), cvScalar(150, 150, 150), 3, CV_AA, 0);
						eye++;      //눈이 몇번 인식이 되었는지 카운트
					}//눈을 인식하기위한 부분



					cvReleaseImage(&fimg);      // 메모리에 로드된 영상이 메모리로부터 해제하기 위해 사용하는 함수이다.
				}
			}
			*/
			cvShowImage("cam", image);      //함수에 의해 지정된 윈도우에 영상을 출력하기 위해 사용하는 함수이다.
			
			if (cvWaitKey(30) >= 0)
				//break;
		//}
		//printf("%d,%d,%d\n", eye, nose, mouth);      //눈, 코, 입이 몇번이나 인식이 되었는지 확인하기 위한 출력문
													 //---------------------------------------------------------------------------------------------------------

		
		/*
		if (eye >= 10 || mouth >= 10 || nose >= 10)      //눈이 10번이상 출력이 되거나 입이 10번이상 코가 10번 이상 출력이 되었다면 모션인식으로 넘어간다.
		{
			*/
			namedWindow("input", 0);
			cvResizeWindow("input", 600, 350);
			//namedWindow("output", 0);
			//namedWindow("threshold", 0);
			//namedWindow("morphology", 0);

			

			//cvDestroyWindow("cam");
			short code = 0;
			code = GetKeyState(VK_SHIFT);
			if (code < 0)
			{
				breakcount++;
			}
			facecount = 21;
			cvtColor(input, output, CV_BGR2HSV, 1);    //RGB->HSV로 변환 (입력 영상, 출력 영상, RGB->HSV로 변환
			inRange(output, hsv_min, hsv_max, img_thresh);

			threshold(img_thresh, binary, 100, 255, THRESH_OTSU);
			//피부색이 min이상 max이하인 색상에 대한 마스크를 만들어 마스크에 저장해서 돌려준다. (예를들어 RGB세개를 한꺼번에 비교하는 것이 아니라 R이 범위에 들어가는지 G가 범위에 들어가는지 B가 범위에 들어가는지를 각각 따로 비교해서 모두 범위에 들어가는 것만 대상이된다.)
			morphology = binary.clone();  //마스크에 대한 클론을 하나 만들고 그 값을 모폴로지에 저장한다.

			for (int i = 0; i < 9; i++)
			{
				erode(morphology, morphology, cv::Mat());  //침식
			}
			for (int i = 0; i < 8; i++)
			{
				dilate(morphology, morphology, cv::Mat());  //팽창
			}

			IplImage *Laveling = new IplImage(morphology);
			IplImage *Lavel = new IplImage(input);

			CBlobLabeling blob;
			blob.SetParam(Laveling, 12000);// 레이블링할 이미지 최소 픽셀수
			blob.DoLabeling();// 레이블링 실행

			if (1)  // m_nNoisze이벤트 생성후 실행시키게 할 수 있음
			{
				int nMaxWidth = Laveling->width * 6 / 10;
				int nMaxHeight = Laveling->height + 6 / 10;
				blob.BlobSmallSizeConstraint(165, 165);
				blob.BlobBigSizeConstraint(nMaxWidth, nMaxHeight);
			}

			if (count >= 0 && count <= 100)
			{
				for (int i = 0; i < blob.m_nBlobs; i++)// int m_nBlobs;레이블의 개수
				{
					CvPoint pt1 = cvPoint(blob.m_recBlobs[i].x, blob.m_recBlobs[i].y); //m_recBlobs; 
					CvPoint pt2 = cvPoint(pt1.x + blob.m_recBlobs[i].width, pt1.y + blob.m_recBlobs[i].height);
					CvScalar color = cvScalar(0, 0, 255); // B,G,R // 빨간색
					cvDrawRect(Lavel, pt1, pt2, color, 2); // 사각형 그리기, 원본, 시작점, 끝점,칼라, 선 두께
				}
				count++;
				if (blob.m_nBlobs == 0)
				{
					counta++;
					if (counta == 100)
					{
						counta = 1;
					}
				}
				if (counta >= 50)
				{
					count = 0;
					counta = 0;
				}
			}
			else if (count > 100)
			{
				start = clock();
				for (int i = 0; i < blob.m_nBlobs; i++)// int m_nBlobs;레이블의 개수
				{
					CvRect hand;
					hand.x = 0; hand.y = 0; hand.width = 0; hand.height = 0;
					hand.x = blob.m_recBlobs[i].x;
					hand.y = blob.m_recBlobs[i].y;
					hand.width = blob.m_recBlobs[i].width;
					hand.height = blob.m_recBlobs[i].height;

					if (hand.width < 170 && hand.height > 190) {
						hand.height = 190;
					}
					if (hand.width > 170 && hand.height > 190) {
						hand.height = 190;
					}


					cvSetImageROI(Ipl_skin, hand);
					IplImage *sub_skin = cvCreateImage(cvSize(hand.width, hand.height), 8, 1);
					cvCopy(Ipl_skin, sub_skin, 0);
					cvResetImageROI(Ipl_skin);

					cvDilate(sub_skin, sub_skin, 0, 1);
					cvErode(sub_skin, sub_skin, 0, 1);


					centerPoints cen;

					CvPoint pt1 = cvPoint(hand.x, hand.y); //m_recBlobs; 
					CvPoint pt2 = cvPoint(pt1.x + hand.width, pt1.y + hand.height);


					// green
					CvScalar color = cvScalar(0, 0, 255); // B,G,R //빨강
					cvDrawRect(Lavel, pt1, pt2, color, 2); // 사각형 그리기, 원본, 시작점, 끝점,칼라, 선 두께

					if (blob.m_nBlobs == 1)
					{
						// green
						CvScalar color = cvScalar(0, 255, 0); // B,G,R //초록
						cvDrawRect(Lavel, pt1, pt2, color, 2); // 사각형 그리기, 원본, 시작점, 끝점,칼라, 선 두께
					}

					if (blob.m_nBlobs >= 2)
					{
						CvPoint new_pt1 = cvPoint(blob.m_recBlobs[i + 1].x, blob.m_recBlobs[i + 1].y); //m_recBlobs;
						CvPoint new_pt2 = cvPoint(new_pt1.x + blob.m_recBlobs[i + 1].width, new_pt1.y + blob.m_recBlobs[i + 1].height);

						// green
						CvScalar new_color = cvScalar(0, 0, 255); // B,G,R //빨강
						cvDrawRect(Lavel, new_pt1, new_pt2, new_color, 2); // 사각형 그리기, 원본, 시작점, 끝점,칼라, 선 두께
					}


					uchar* data_sub = (uchar*)sub_skin->imageData;
					int sub_w = sub_skin->width;
					int sub_h = sub_skin->height;
					int sub_ws = sub_skin->widthStep;

					for (int j = 0; j < sub_h; j++)
						for (int i = 0; i < sub_w; i++) {
							if (data_sub[j*sub_ws + i] == 255) {
								cp.x[cp.num] = i;
								cp.y[cp.num] = j;
								cp.num++;
							}
						}
					printf("pixel=%d", cp.num);
					for (int k = 0; k < cp.num; k++) {
						sum_x += cp.x[k];
						sum_y += cp.y[k];
					}

					for (int j = 0; j < sub_h; j++)
						for (int i = 0; i < sub_w; i++)
						{
							if (data_sub[j*sub_ws + i] == 255)
							{
								fst.x = i;
								fst.y = j;
								i = sub_w; j = sub_h;
							}
						}

					cvCircle(Lavel, cvPoint(pt1.x + (int)(sum_x / cp.num), pt1.y + (int)(sum_y / cp.num)), 5, CV_RGB(0, 255, 0), 5);
					//circle(input, Point(pt1.x + (int)(sum_x / cp.num), pt1.y + (int)(sum_y / cp.num)), 5, CV_RGB(0, 255, 0), 5);

					printf("x=%d,y=%d\n", (sum_x / cp.num), (sum_y / cp.num));      //중심점의 좌표 출력
					countlocation++;
					countall++;

					


					
					int rad = (int)sqrt((double)(((int)(sum_x / cp.num) - fst.x) * ((int)(sum_x / cp.num) - fst.x) + (((int)(sum_y / cp.num) - fst.y)*((int)(sum_y / cp.num) - fst.y))));
					cvCircle(Lavel, cvPoint(pt1.x + (int)(sum_x / cp.num), pt1.y + (int)(sum_y / cp.num)), (int)(rad / 1.5), CV_RGB(0, 255, 0), 6);
					cvLine(Lavel, cvPoint(pt1.x + fst.x, pt1.y + fst.y), cvPoint(pt1.x + (int)(sum_x / cp.num), pt1.y + (int)(sum_y / cp.num)), CV_RGB(255, 255, 0), 4);

					cen.center_x = pt1.x + (int)(sum_x / cp.num);
					cen.center_y = pt1.y + (int)(sum_y / cp.num);

					average_x = point_s.x - cen.center_x;
					average_y = cen.center_y - point_s.y;

					for (int i = 0; i <= 6; i++) {
						average_x += point_s.x - cen.center_x;
						average_y += cen.center_y - point_s.y;
					}

					average_x = average_x / 6;
					average_y = average_y / 6;

					SetCursorPos((int)(point_i.x + 2 * (average_x)* 800. / Laveling->width), (int)(point_i.y + 2 * (average_y)*680. / Laveling->height));

					point_s.x = cen.center_x;
					point_s.y = cen.center_y;

					GetCursorPos(&point_i);

					if (((int)(point_i.y) <= (total_height)) && ((int)(point_i.x) <= (total_width))     )
					{
						view = 1;
					}

					else if (((int)(point_i.y) <= (total_height)) && ((int)(point_i.x) >= (total_width)))
					{
						view = 2;
					}
					else if (((int)(point_i.y) > (total_height)) && ((int)(point_i.x) < (total_width)))
					{
						view = 3;
					}

					else if (((int)(point_i.y) > (total_height)) && ((int)(point_i.x) > (total_width)))
					{
						view = 4;
					}

					IplImage* circle = cvCreateImage(cvGetSize(sub_skin), 8, 1);
					cvSetZero(circle);
					cvCircle(circle, cvPoint((int)(sum_x / cp.num), (int)(sum_y / cp.num)), (int)(rad / 1.5), CV_RGB(255, 255, 255), 6);
					cvAnd(sub_skin, circle, sub_skin, 0);

					if (countlocation == 4)
					{
						prevX = (sum_x / cp.num);
						prevY = (sum_y / cp.num);
					}
					printf("\tprevX=%d, prevY=%d\n", prevX, prevY);

					if (((prevX >= ((sum_x / cp.num) - 2)) && (prevX <= ((sum_x / cp.num) + 2))) && ((prevY >= ((sum_y / cp.num) - 2)) && (prevY <= ((sum_y / cp.num) + 2))))
					{
						printf("prevX=%d, prevY=%d", prevX, prevY);
						printf("!!!!!!!!!!!!!!!!true\n");
						locationpoint = locationpoint + locationvalue;
					}
					if (countlocation == 5)
					{
						countlocation = 0;
					}

					if (countall == 20)
					{
						short clickcode = 0;
						clickcode = GetKeyState(VK_LCONTROL);
						if (clickcode < 0)
						{
							clickcount++;
						}
						if (clickcode == 0)
						{

							if (locationpoint >= 7)
							{
								printf("!!!!!!!!!!!!!!!!!!!!!!!!down!!!!!!!!!!!!!!!!!!!!\n");
								mouse_event(MOUSEEVENTF_LEFTDOWN, (sum_x / cp.num), (sum_y / cp.num), 0, GetMessageExtraInfo());
								mouseup_count++;



							}
							if (mouseup_count >= 1)
							{
								if (((prevX >= ((sum_x / cp.num) - 2)) && (prevX <= ((sum_x / cp.num) + 2))) && ((prevY >= ((sum_y / cp.num) - 2)) && (prevY <= ((sum_y / cp.num) + 2))))
								{
									printf("!!!!!!!!!!!!!!!!!!!!!!!!up!!!!!!!!!!!!!!!!!!!!\n");
									mouse_event(MOUSEEVENTF_LEFTUP, (sum_x / cp.num), (sum_y / cp.num), 0, GetMessageExtraInfo());

									mouseup_count = 0;
									finish = clock();
									duration = (double)(finish - start) / CLOCKS_PER_SEC;
									f = fopen("a.txt", "w");
									fprintf(f, "view = %d\n", view);
									printf("click View : %d\n", view);
									printf("%d, %d\n", (int)(point_i.x + 2 * (average_x)* 800), (int)(point_i.y + 2 * (average_y)* 800));
									fprintf(f, "%f 초\n", duration);
									fclose(f);

								}
							}
							short clickcode = 0;
							clickcode = GetKeyState(VK_LCONTROL);

							//printf("duration = %d", duration);
							if (clickcode < 0)
							{
								clickcount++;
							}
						}
						/*if (clickcode >= 1)
						{
						if (locationpoint >= 7)
						{
						printf("!!!!!!!!!!!!!!!!!!!!!!!!click!!!!!!!!!!!!!!!!!!!!\n");
						mouse_event(MOUSEEVENTF_LEFTDOWN, (sum_x / cp.num), (sum_y / cp.num), 0, GetMessageExtraInfo());
						mouse_event(MOUSEEVENTF_LEFTUP, (sum_x / cp.num), (sum_y / cp.num), 0, GetMessageExtraInfo());
						}
						short clickcode = 0;
						clickcode = GetKeyState(VK_LCONTROL);
						if (clickcode < 0)
						{
						clickcount=0;
						}
						}*/



						printf("locationpoint=%d\n", locationpoint);
						locationpoint = 0;
						countall = 0;
					}

					
				}
				count = 101;
				if (blob.m_nBlobs == 0)         //레이블링의 개수가 0일때
				{
					counta++;
					if (counta == 2000)         //일정이상 카운트가 증가하면 줄여준다.
					{
						counta = 1;
					}
				}
			}
			imshow("input", input);
			//imshow("output", output);
			//imshow("threshold", binary);
			//imshow("morphology", morphology);
		}
		waitKey(30);
	}

