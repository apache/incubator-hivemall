package hivemall.lda;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;


public class Test_OnlineLDA_Formatted{
	// Container
	static ArrayList<String> fileNames = new ArrayList<String>();
	static ArrayList<String> topicGroup= new ArrayList<String>();
	static ArrayList<Integer> gammaTopics= new ArrayList<Integer>();
	static ArrayList<String[][]> stringBatchList = new ArrayList<String[][]>();
	static HashMap<String, ArrayList<Integer>> topicGammaTopicMap = new HashMap<String, ArrayList<Integer>>();

	// Constant Parameters
	static int batchSize_ = 10;
	static String targetURI;
	
	// limit Read File Per Directory
//	static int limit = 10;

	// LDA Parameters
	static int K = 6;
	static double alpha = 1./(K);
	static double eta = 1./ (K);
	static double tau0 = 80;	// 1024
	static double kappa = 0.8;	// 0.7
	static int IterNum = 1;
	static int PPLNUM = 10;
//	static int totalD= (int)11000;
	static int totalD= 11000 * PPLNUM;

	// Control
	static int limit = 10000;
	static int trainLine=10000;
	
	static OnlineLDAModel model;

//	static String stopWord = "a b c d e f g h i j k l m n o p q r s t u v w x y z the of in and have to it was or were this that with is some on for so how you if would com be your my one not never then take for an can no but aaa when as out just from does they back up she those who another her do by must what there at very are am much way all any other me he something someone doesn his also its has into us him than about their may too will had been we them why did being over without these could out which only should even well more where after while anyone our now such under two ten else always going either each however non let done ever between anything before every same since because quite sure here nothing new don off still down yes around few many own go get know think like make say see look use said";
	static String stopWord = "a ,able ,about ,above ,abst ,accordance ,according ,accordingly ,across ,act ,actually ,added ,adj ,affected ,affecting ,affects ,after ,afterwards ,again ,against ,ah ,all ,almost ,alone ,along ,already ,also ,although ,always ,am ,among ,amongst ,an ,and ,announce ,another ,any ,anybody ,anyhow ,anymore ,anyone ,anything ,anyway ,anyways ,anywhere ,apparently ,approximately ,are ,aren ,arent ,arise ,around ,as ,aside ,ask ,asking ,at ,auth ,available ,away ,awfully ,b ,back ,be ,became ,because ,become ,becomes ,becoming ,been ,before ,beforehand ,begin ,beginning ,beginnings ,begins ,behind ,being ,believe ,below ,beside ,besides ,between ,beyond ,biol ,both ,brief ,briefly ,but ,by ,c ,ca ,came ,can ,cannot ,can't ,cause ,causes ,certain ,certainly ,co ,com ,come ,comes ,contain ,containing ,contains ,could ,couldnt ,d ,date ,did ,didn't ,different ,do ,does ,doesn't ,doing ,done ,don't ,down ,downwards ,due ,during ,e ,each ,ed ,edu ,effect ,eg ,eight ,eighty ,either ,else ,elsewhere ,end ,ending ,enough ,especially ,et ,et-al ,etc ,even ,ever ,every ,everybody ,everyone ,everything ,everywhere ,ex ,except ,f ,far ,few ,ff ,fifth ,first ,five ,fix ,followed ,following ,follows ,for ,former ,formerly ,forth ,found ,four ,from ,further ,furthermore ,g ,gave ,get ,gets ,getting ,give ,given ,gives ,giving ,go ,goes ,gone ,got ,gotten ,h ,had ,happens ,hardly ,has ,hasn't ,have ,haven't ,having ,he ,hed ,hence ,her ,here ,hereafter ,hereby ,herein ,heres ,hereupon ,hers ,herself ,hes ,hi ,hid ,him ,himself ,his ,hither ,home ,how ,howbeit ,however ,hundred ,i ,id ,ie ,if ,i'll ,im ,immediate ,immediately ,importance ,important ,in ,inc ,indeed ,index ,information ,instead ,into ,invention ,inward ,is ,isn't ,it ,itd ,it'll ,its ,itself ,i've ,j ,just ,k ,keep, keeps ,kept ,kg ,km ,know ,known ,knows ,l ,largely ,last ,lately ,later ,latter ,latterly ,least ,less ,lest ,let ,lets ,like ,liked ,likely ,line ,little ,'ll ,look ,looking ,looks ,ltd ,m ,made ,mainly ,make ,makes ,many ,may ,maybe ,me ,mean ,means ,meantime ,meanwhile ,merely ,mg ,might ,million ,miss ,ml ,more ,moreover ,most ,mostly ,mr ,mrs ,much ,mug ,must ,my ,myself ,n ,na ,name ,namely ,nay ,nd ,near ,nearly ,necessarily ,necessary ,need ,needs ,neither ,never ,nevertheless ,new ,next ,nine ,ninety ,no ,nobody ,non ,none ,nonetheless ,noone ,nor ,normally ,nos ,not ,noted ,nothing ,now ,nowhere ,o ,obtain ,obtained ,obviously ,of ,off ,often ,oh ,ok ,okay ,old ,omitted ,on ,once ,one ,ones ,only ,onto ,or ,ord ,other ,others ,otherwise ,ought ,our ,ours ,ourselves ,out ,outside ,over ,overall ,owing ,own ,p ,page ,pages ,part ,particular ,particularly ,past ,per ,perhaps ,placed ,please ,plus ,poorly ,possible ,possibly ,potentially ,pp ,predominantly ,present ,previously ,primarily ,probably ,promptly ,proud ,provides ,put ,q ,que ,quickly ,quite ,qv ,r ,ran ,rather ,rd ,re ,readily ,really ,recent ,recently ,ref ,refs ,regarding ,regardless ,regards ,related ,relatively ,research ,respectively ,resulted ,resulting ,results ,right ,run ,s ,said ,same ,saw ,say ,saying ,says ,sec ,section ,see ,seeing ,seem ,seemed ,seeming ,seems ,seen ,self ,selves ,sent ,seven ,several ,shall ,she ,shed ,she'll ,shes ,should ,shouldn't ,show ,showed ,shown ,showns ,shows ,significant ,significantly ,similar ,similarly ,since ,six ,slightly ,so ,some ,somebody ,somehow ,someone ,somethan ,something ,sometime ,sometimes ,somewhat ,somewhere ,soon ,sorry ,specifically ,specified ,specify ,specifying ,still ,stop ,strongly ,sub ,substantially ,successfully ,such ,sufficiently ,suggest ,sup ,sure	t ,take ,taken ,taking ,tell ,tends ,th ,than ,thank ,thanks ,thanx ,that ,that'll ,thats ,that've ,the ,their ,theirs ,them ,themselves ,then ,thence ,there ,thereafter ,thereby ,thered ,therefore ,therein ,there'll ,thereof ,therere ,theres ,thereto ,thereupon ,there've ,these ,they ,theyd ,they'll ,theyre ,they've ,think ,this ,those ,thou ,though ,thoughh ,thousand ,throug ,through ,throughout ,thru ,thus ,til ,tip ,to ,together ,too ,took ,toward ,towards ,tried ,tries ,truly ,try ,trying ,ts ,twice ,two ,u ,un ,under ,unfortunately ,unless ,unlike ,unlikely ,until ,unto ,up ,upon ,ups ,us ,use ,used ,useful ,usefully ,usefulness ,uses ,using ,usually ,v ,value ,various ,'ve ,very ,via ,viz ,vol ,vols ,vs ,w ,want ,wants ,was ,wasnt ,way ,we ,wed ,welcome ,we'll ,went ,were ,werent ,we've ,what ,whatever ,what'll ,whats ,when ,whence ,whenever ,where ,whereafter ,whereas ,whereby ,wherein ,wheres ,whereupon ,wherever ,whether ,which ,while ,whim ,whither ,who ,whod ,whoever ,whole ,who'll ,whom ,whomever ,whos ,whose ,why ,widely ,willing ,wish ,with ,within ,without ,wont ,words ,world ,would ,wouldnt ,www ,x ,y ,yes ,yet ,you ,youd ,you'll ,your ,youre ,yours ,yourself ,yourselves ,you've ,z ,zero , stopwords_en.txt ,a ,a's ,able ,about ,above ,according ,accordingly ,across ,actually ,after ,afterwards ,again ,against ,ain't ,all ,allow ,allows ,almost ,alone ,along ,already ,also ,although ,always ,am ,among ,amongst ,an ,and ,another ,any ,anybody ,anyhow ,anyone ,anything ,anyway ,anyways ,anywhere ,apart ,appear ,appreciate ,appropriate ,are ,aren't ,around ,as ,aside ,ask ,asking ,associated ,at ,available ,away ,awfully ,b ,be ,became ,because ,become ,becomes ,becoming ,been ,before ,beforehand ,behind ,being ,believe ,below ,beside ,besides ,best ,better ,between ,beyond ,both ,brief ,but ,by ,c ,c'mon ,c's ,came ,can ,can't ,cannot ,cant ,cause ,causes ,certain ,certainly ,changes ,clearly ,co ,com ,come ,comes ,concerning ,consequently ,consider ,considering ,contain ,containing ,contains ,corresponding ,could ,couldn't ,course ,currently ,d ,definitely ,described ,despite ,did ,didn't ,different ,do ,does ,doesn't ,doing ,don't ,done ,down ,downwards ,during ,e ,each ,edu ,eg ,eight ,either ,else ,elsewhere ,enough ,entirely ,especially ,et ,etc ,even ,ever ,every ,everybody ,everyone ,everything ,everywhere ,ex ,exactly ,example ,except ,f ,far ,few ,fifth ,first ,five ,followed ,following ,follows ,for ,former ,formerly ,forth ,four ,from ,further ,furthermore ,g ,get ,gets ,getting ,given ,gives ,go ,goes ,going ,gone ,got ,gotten ,greetings ,h ,had ,hadn't ,happens ,hardly ,has ,hasn't ,have ,haven't ,having ,he ,he's ,hello ,help ,hence ,her ,here ,here's ,hereafter ,hereby ,herein ,hereupon ,hers ,herself ,hi ,him ,himself ,his ,hither ,hopefully ,how ,howbeit ,however ,i ,i'd ,i'll ,i'm ,i've ,ie ,if ,ignored ,immediate ,in ,inasmuch ,inc ,indeed ,indicate ,indicated ,indicates ,inner ,insofar ,instead ,into ,inward ,is ,isn't ,it ,it'd ,it'll ,it's ,its ,itself ,j ,just ,k ,keep ,keeps ,kept ,know ,knows ,known ,l ,last ,lately ,later ,latter ,latterly ,least ,less ,lest ,let ,let's ,like ,liked ,likely ,little ,look ,looking ,looks ,ltd ,m ,mainly ,many ,may ,maybe ,me ,mean ,meanwhile ,merely ,might ,more ,moreover ,most ,mostly ,much ,must ,my ,myself ,n ,name ,namely ,nd ,near ,nearly ,necessary ,need ,needs ,neither ,never ,nevertheless ,new ,next ,nine ,no ,nobody ,non ,none ,noone ,nor ,normally ,not ,nothing ,novel ,now ,nowhere ,o ,obviously ,of ,off ,often ,oh ,ok ,okay ,old ,on ,once ,one ,ones ,only ,onto ,or ,other ,others ,otherwise ,ought ,our ,ours ,ourselves ,out ,outside ,over ,overall ,own ,p ,particular ,particularly ,per ,perhaps ,placed ,please ,plus ,possible ,presumably ,probably ,provides ,q ,que ,quite ,qv ,r ,rather ,rd ,re ,really ,reasonably ,regarding ,regardless ,regards ,relatively ,respectively ,right ,s ,said ,same ,saw ,say ,saying ,says ,second ,secondly ,see ,seeing ,seem ,seemed ,seeming ,seems ,seen ,self ,selves ,sensible ,sent ,serious ,seriously ,seven ,several ,shall ,she ,should ,shouldn't ,since ,six ,so ,some ,somebody ,somehow ,someone ,something ,sometime ,sometimes ,somewhat ,somewhere ,soon ,sorry ,specified ,specify ,specifying ,still ,sub ,such ,sup ,sure ,t ,t's ,take ,taken ,tell ,tends ,th ,than ,thank ,thanks ,thanx ,that ,that's ,thats ,the ,their ,theirs ,them ,themselves ,then ,thence ,there ,there's ,thereafter ,thereby ,therefore ,therein ,theres ,thereupon ,these ,they ,they'd ,they'll ,they're ,they've ,think ,third ,this ,thorough ,thoroughly ,those ,though ,three ,through ,throughout ,thru ,thus ,to ,together ,too ,took ,toward ,towards ,tried ,tries ,truly ,try ,trying ,twice ,two ,u ,un ,under ,unfortunately ,unless ,unlikely ,until ,unto ,up ,upon ,us ,use ,used ,useful ,uses ,using ,usually ,uucp ,v ,value ,various ,very ,via ,viz ,vs ,w ,want ,wants ,was ,wasn't ,way ,we ,we'd ,we'll ,we're ,we've ,welcome ,well ,went ,were ,weren't ,what ,what's ,whatever ,when ,whence ,whenever ,where ,where's ,whereafter ,whereas ,whereby ,wherein ,whereupon ,wherever ,whether ,which ,while ,whither ,who ,who's ,whoever ,whole ,whom ,whose ,why ,will ,willing ,wish ,with ,within ,without ,won't ,wonder ,would ,would ,wouldn't ,x ,y ,yes ,yet ,you ,you'd ,you'll ,you're ,you've ,your ,yours ,yourself ,yourselves ,z ,zero";
	
	public static void main(String[] args){
		long start = System.nanoTime();
//		targetURI = "/Users/ishikawanaoki/dataset/news20_tdIdf.txt";
		targetURI = "/Users/ishikawanaoki/dataset/reuters_tfidf.txt";
		
		model = new OnlineLDAModel(K, alpha, eta, totalD, tau0, kappa, batchSize_, stopWord);

		try {
			executeTraining();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		long end = System.nanoTime();
		
//		printConfusionMatrix();
		model.showTopicWords();
		
		System.out.println("Experiment Time:" + (end - start));
	}

	private static void executeTraining() throws IOException {

		int time = 0;


		BufferedReader br;
		for(int iter=0; iter<PPLNUM; iter++){
			System.out.println("Iteration:" + iter);
			br = new BufferedReader(new FileReader(targetURI));

			ArrayList<String[]> miniBatchArrayList = new ArrayList<String[]>();

			while(true){
				String docString = br.readLine();
				time++;

				if(docString == null){
					executeMiniBatchLearining(miniBatchArrayList, time);
					break;
				}else{
					String[] labelValues = docString.split(" "); 
					miniBatchArrayList.add(labelValues);

					if(miniBatchArrayList.size() == batchSize_){
						executeMiniBatchLearining(miniBatchArrayList, time);
						miniBatchArrayList.clear();
					}
				}
			}
			System.out.print("," + model.getPerplexity());
		}
	}
	



	private static void executeMiniBatchLearining(ArrayList<String[]> miniBatchArrayList, int time) {
		String[][] miniBatch = new String[miniBatchArrayList.size()][];
		for(int d=0, SIZE = miniBatchArrayList.size(); d< SIZE; d++){
			miniBatch[d] = miniBatchArrayList.get(d);
		}
		
		model.train(miniBatch, time);
	}

//	private static String[] processLine(String line) {
//		
//		String[] stopWords = stopWord.split(" ");
//		HashSet<String> stopWordSet = new HashSet<String>();
//		for(String tmpStopWord: stopWords){
//			stopWordSet.add(tmpStopWord);
//		}
//
//		line = line.toLowerCase();
//		
//		line = line.replace("\"", " ");
//		line = line.replace("\\", " ");
//		line = line.replace("/", " ");
//		line = line.replace(">", " ");
//		line = line.replace("<", " ");
//		line = line.replace("-", " ");
//		line = line.replace(",", " ");
//		line = line.replace(".", " ");
//		line = line.replace("(", " ");
//		line = line.replace(")", " ");
//		line = line.replace(":", " ");
//		line = line.replace(";", " ");
//		line = line.replace("'", " ");
//		line = line.replace("[", " ");
//		line = line.replace("]", " ");
//		line = line.replace("!", " ");
//		line = line.replace("*", " ");
//		line = line.replace("#", " ");
//		line = line.replace("+", " ");
//		line = line.replace("%", " ");
//		line = line.replace("@", " ");
//		line = line.replace("&", " ");
//		line = line.replace("?", " ");
//		line = line.replace("$", " ");
//		line = line.replace("0", " ");
//		line = line.replace("1", " ");
//		line = line.replace("2", " ");
//		line = line.replace("3", " ");
//		line = line.replace("4", " ");
//		line = line.replace("5", " ");
//		line = line.replace("6", " ");
//		line = line.replace("7", " ");
//		line = line.replace("8", " ");
//		line = line.replace("9", " ");
//		line = line.replace("\t", " ");
//		line = line.replace("_", " ");
//		line = line.replace("{", " ");
//		line = line.replace("}", " ");
//		line = line.replace("=", " ");
//		line = line.replace("|", " ");
//		
//		
//		String[] ret;
//		
//		
//		ret = line.split(" ");
//		
//		if(line.equals("")){
//			ret = new String[1];
//			ret[0] = "XXXXXXXXXXXXXXX";
//			return ret;
//		}
//		
//		ArrayList<String> tmpArrayList = new ArrayList<String>();
//		for(int i=0; i<ret.length; i++){
//			if(ret[i].length() >= 0 && !stopWordSet.contains(ret[i])){
//				tmpArrayList.add(ret[i]);
//			}
//		}
//		
//		String[] ret2 = new String[tmpArrayList.size()];
//		for(int i=0, Size=tmpArrayList.size(); i<Size; i++){
//			ret2[i] = tmpArrayList.get(i);
//		}
//
//		return ret2;
//	}
}
