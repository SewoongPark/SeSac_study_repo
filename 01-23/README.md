## **TIL**
### **FLASK 사용한 크롤링**
**여러가지 url 주소와 html rendering하기**
* index.html과 index_table 만들기
* `<table>`과 `<thead>` `<tbody>` `<tr>` 구조 이해
* `<button>`과 `<a href>` 활용하여 두 페이지 간 연결 

**html문서 dictionary로 변경하고 rendering**


### **데이터 베이스**
* **시스템**
* 시스템 데이터셋의 구조
    * 예) 고객 데이터 베이스(DB)
    * 고객 신상 정보 DB: 성별, 나이, 취미, 결혼 유무
    * 구매 DB, 상품 DB, 구입 거래처 DB
* **RDMBS**
* https://blog.siner.io/2021/10/11/rdbms-comparison/
* 주요 RDBMS의 종류:<table> <tr> <td> mysql</td> <td> mariadb<td> <td>postgresql</td> <td>sqlite</td> <td> oracle </td> <td>mssql<td> </tr> </table>

**Structured Query Language : 규격화된, 규칙을 정해놓은 질의 언어**

<img src = https://velog.velcdn.com/images/jude0124/post/aad18652-306f-4b7a-b16e-fd74428978c3/image.png style="width: 30%;">

**이러한 SQL 명령어의 종류는 크게 아래와 같이 다섯가지로 나뉩니다.**

🔴 DQL(질의어)

🟠 DML(데이터조작어)

🟡 DDL(데이터정의어)

🟢 TCL(트랜잭션처리어)

🔵 DCL(데이터제어어)


**MYSQL**
<ul>
<li>top n개의 레코드를 가지고 오는 케이스에 특화되어있다</li>
<li>웹 애플리케이션으로서의 MySQL의 인기는 PHP의 인기도와 맞물려있다</li>
<li>복잡한 알고리즘은 가급적 지원하지 않는다</li>
<li>간단한 처리속도를 향상시키는 것을 추구한다</li>
<li>정확함</li>
</ul>

**SQL 서버 2005**
* 기능과 이용자에 따라 여러 버전으로 나누어 배포하고 있다. 이에 따른 버전은 다음과 같다
<ul>
<li> SQL 서버 콤팩트 에디션 (SQL CE)</li>
<li> SQL 서버 익스프레스 에디션</li>
<li> SQL 서버 워크그룹 에디션</li>
<li> SQL 서버 스탠다드 에디션</li>
<li> SQL 서버 엔터프라이즈 에디션</li>
<li> SQL 서버 디벨로퍼 에디션</li>
<li> SQL 서버 웹 에디션</li>
</ul>

### **서브 쿼리(하위 쿼리)**
**서브 쿼리란?**
* 하나의 쿼리문 안에 포함되어 있는 또 하나의 쿼리문을 말합니다.
* 서브 쿼리는 메인 쿼리가 서브 쿼리를 포함하는 종속적인 관계입니다. 
* 여러 번의 쿼리를 수행해야만 얻을 수 있는 결과를 하나의 중첩된 SQL 문장으로 간편하게 결과를 얻을 수 있게 해 줍니다.

주의사항

* ☞ 서브 쿼리를 괄호로 묶어서 사용해야 한다.
* ☞ 서브 쿼리 안에서 Order By 절은 사용할 수 없다.
* ☞ 연산자 오른쪽에 사용하여야 한다.

```sql
SELECT
    CD_PLANT,
    NM_PLANT,
(SELECT AVG(UM) FROM TABLE_ITEM) AS UM
FROM TABLE_PLANT
```
