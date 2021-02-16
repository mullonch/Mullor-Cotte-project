import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import server
from flask import json


def test_say_hello():
    response = server.test_client().get('/hello')

    assert response.status_code == 200
    assert response.data == b'Welcome to the real article classifier !'


def test_predict_fake_news():
    response = server.test_client().post(
        '/predict',
        data=json.dumps({
            "title": "Tone Deaf Trump: Congrats Rep. Scalise On Losing Weight After You Almost Died",
            "date": "January 14, 2016",
            "text": "Donald Trump just signed the GOP tax scam into law. Of course, that meant that he invited all of "
                    "his craven, cruel GOP sycophants down from their perches on Capitol Hill to celebrate in the Rose "
                    "Garden at the White House. Now, that part is bad enough   celebrating tax cuts for a bunch of rich "
                    "hedge fund managers and huge corporations at the expense of everyday Americans. Of course, Trump is "
                    "beside himself with glee, as this represents his first major legislative win since he started squatting "
                    "in the White House almost a year ago. Thanks to said glee, in true Trumpian style, he gave a free-wheeling "
                    "address, and a most curious subject came up as Trump was thanking the goons from the Hill. Somehow, "
                    "Trump veered away from tax cuts, and started talking about the Congressional baseball shooting that "
                    "happened over the summer.In that shooting, Rep. Steve Scalise, who is also the House Majority Whip, "
                    "was shot and almost lost his life. Thanks to this tragic and stunning act of political violence, Scalise "
                    "had a long recovery  in fact he is still in physical therapy. But, of course, vain and looks-obsessed "
                    "Trump decided that he would congratulate Scalise, not on his survival and on his miraculous recovery, "
                    "but on the massive amount of weight Scalise lost while he was practically dying. And make no mistake   "
                    "Scalise is VERY lucky to be alive. According to doctors, when he arrived at the hospital, Scalise was "
                    "actually, quote, in  imminent risk of death.  Here is the quote, via Twitter:How stunningly tone deaf "
                    "does one have to be to say something like that? I never thought I d say this about a Republican that "
                    "I, by all reasonable accounts, absolutely loathe, but I feel sorry for him. I am sorry he got shot, "
                    "and I am even sorrier that he now has to stand there and listen to that orange buffoon talk about "
                    "him like that.I am sure that Scalise is a much tougher man than Trump, though. I am equally sure that "
                    "he also knows that Trump is an international embarrassment and a crazy man who never should have been "
                    "allowed anywhere near the White House.Featured image via Alex Wong/Getty Images",
            "subject": "Middle-east"}),
        content_type='application/json',
    )

    data = json.loads(response.get_data(as_text=True))
    assert response.status_code == 200
    assert response.data == b'"This is a fake"\n'


def test_predict_real_news():
    response = server.test_client().post(
        '/predict',
        data=json.dumps({
            "title": "As U.S. budget fight looms, Republicans flip their fiscal script",
            "date": "December 31, 2017",
            "text": "WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted "
                    "this month for a huge expansion of the national debt to pay for tax cuts, called himself a “fiscal "
                    "conservative” on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way "
                    "among Republicans, U.S. Representative Mark Meadows, speaking on CBS’ “Face the Nation,” drew a hard "
                    "line on federal spending, which lawmakers are bracing to do battle over in January. When they return "
                    "from the holidays on Wednesday, lawmakers will begin trying to pass a federal budget in a fight likely "
                    "to be linked to other issues, such as immigration policy, even as the November congressional election "
                    "campaigns approach in which Republicans will seek to keep control of Congress. President Donald Trump "
                    "and his Republicans want a big budget increase in military spending, while Democrats also want proportional "
                    "increases for non-defense “discretionary” spending on programs that support education, scientific research, "
                    "infrastructure, public health and environmental protection. “The (Trump) administration has already "
                    "been willing to say: ‘We’re going to increase non-defense discretionary spending ... by about 7 percent,’” "
                    "Meadows, chairman of the small but influential House Freedom Caucus, said on the program. “Now, Democrats "
                    "are saying that’s not enough, we need to give the government a pay raise of 10 to 11 percent. For a "
                    "fiscal conservative, I don’t see where the rationale is. ... Eventually you run out of other people’s "
                    "money,” he said. Meadows was among Republicans who voted in late December for their party’s debt-financed "
                    "tax overhaul, which is expected to balloon the federal budget deficit and add about $1.5 trillion "
                    "over 10 years to the $20 trillion national debt. “It’s interesting to hear Mark talk about fiscal "
                    "responsibility,” Democratic U.S. Representative Joseph Crowley said on CBS. Crowley said the Republican "
                    "tax bill would require the United States to borrow $1.5 trillion, to be paid off by future generations, "
                    "to finance tax cuts for corporations and the rich. “This is one of the least ... fiscally responsible "
                    "bills we’ve ever seen passed in the history of the House of Representatives. I think we’re going to "
                    "be paying for this for many, many years to come,” Crowley said. Republicans insist the tax package, "
                    "the biggest U.S. tax overhaul in more than 30 years, will boost the economy and job growth. House Speaker "
                    "Paul Ryan, who also supported the tax bill, recently went further than Meadows, making clear in a radio "
                    "interview that welfare or “entitlement reform,” as the party often calls it, would be a top Republican "
                    "priority in 2018. In Republican parlance, “entitlement” programs mean food stamps, housing assistance, "
                    "Medicare and Medicaid health insurance for the elderly, poor and disabled, as well as other programs "
                    "created by Washington to assist the needy. Democrats seized on Ryan’s early December remarks, saying "
                    "they showed Republicans would try to pay for their tax overhaul by seeking spending cuts for social "
                    "programs. But the goals of House Republicans may have to take a back seat to the Senate, where the "
                    "votes of some Democrats will be needed to approve a budget and prevent a government shutdown. Democrats "
                    "will use their leverage in the Senate, which Republicans narrowly control, to defend both discretionary "
                    "non-defense programs and social spending, while tackling the issue of the “Dreamers,” people brought "
                    "illegally to the country as children. Trump in September put a March 2018 expiration date on the Deferred "
                    "Action for Childhood Arrivals, or DACA, program, which protects the young immigrants from deportation "
                    "and provides them with work permits. The president has said in recent Twitter messages he wants funding "
                    "for his proposed Mexican border wall and other immigration law changes in exchange for agreeing to help "
                    "the Dreamers. Representative Debbie Dingell told CBS she did not favor linking that issue to other "
                    "policy objectives, such as wall funding. “We need to do DACA clean,” she said. On Wednesday, Trump "
                    "aides will meet with congressional leaders to discuss those issues. That will be followed by a weekend "
                    "of strategy sessions for Trump and Republican leaders on Jan. 6 and 7, the White House said. Trump was "
                    "also scheduled to meet on Sunday with Florida Republican Governor Rick Scott, who wants more emergency "
                    "aid. The House has passed an $81 billion aid package after hurricanes in Florida, Texas and Puerto Rico, "
                    "and wildfires in California. The package far exceeded the $44 billion requested by the Trump administration. "
                    "The Senate has not yet voted on the aid.",
            "subject": "politicsNews"}),
        content_type='application/json',
    )

    data = json.loads(response.get_data(as_text=True))
    assert response.status_code == 200
    assert response.data == b'"This is a real news article"\n'


